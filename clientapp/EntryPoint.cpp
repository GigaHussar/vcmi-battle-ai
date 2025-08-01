/*
 * EntryPoint.cpp, part of VCMI engine
 *
 * Authors: listed in file AUTHORS in main folder
 *
 * License: GNU General Public License v2.0 or later
 * Full text of license available in license.txt file, in main folder
 *
 */

// EntryPoint.cpp : Defines the entry point for the console application.

#include "StdInc.h"
#include "../Global.h"

#include "../client/ClientCommandManager.h"
#include "../client/CMT.h"
#include "../client/CPlayerInterface.h"
#include "../client/CServerHandler.h"
#include "../client/GameEngine.h"
#include "../client/GameInstance.h"
#include "../client/gui/CursorHandler.h"
#include "../client/gui/WindowHandler.h"
#include "../client/mainmenu/CMainMenu.h"
#include "../client/render/Graphics.h"
#include "../client/render/IRenderHandler.h"
#include "../client/windows/CMessage.h"
#include "../client/windows/InfoWindows.h"

#include "../lib/AsyncRunner.h"
#include "../lib/CConsoleHandler.h"
#include "../lib/CConfigHandler.h"
#include "../lib/CThreadHelper.h"
#include "../lib/ExceptionsCommon.h"
#include "../lib/filesystem/Filesystem.h"
#include "../lib/logging/CBasicLogConfigurator.h"
#include "../lib/modding/IdentifierStorage.h"
#include "../lib/modding/CModHandler.h"
#include "../lib/modding/ModDescription.h"
#include "../lib/texts/MetaString.h"
#include "../lib/GameLibrary.h"
#include "../lib/VCMIDirs.h"

#include <boost/program_options.hpp>
#include <vstd/StringUtils.h>

#include <SDL_main.h>
#include <SDL.h>
#include <thread>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include "../client/battle/BattleActionsController.h"
#include "../client/battle/BattleInterface.h"
#include "StdInc.h"
#include "GameInstance.h"

// Pull in the adventure‐map state and selection API
#include "../client/PlayerLocalState.h"       // defines PlayerLocalState::getCurrentHero()

// Full definitions for CGHeroInstance and int3
#include "../lib/mapObjects/CGHeroInstance.h"
#include "../lib/pathfinder/CGPathNode.h"      // where `int3` is declared

// The callback interface that packages and sends MoveHero packets
#include "../lib/callback/CCallback.h"

#include "mainmenu/CMainMenu.h"             // for CMainMenu::openLobby
#include "lobby/CSelectionBase.h" // for ESelectionScreen
#include "lobby/CLobbyScreen.h"   // for ELoadMode


#ifdef VCMI_ANDROID
#include "../lib/CAndroidVMHelper.h"
#include <SDL_system.h>
#endif

#if __MINGW32__
#undef main
#endif

namespace po = boost::program_options;
namespace po_style = boost::program_options::command_line_style;

static std::atomic<bool> headlessQuit = false;
static std::optional<std::string> criticalInitializationError;

static void init()
{
	try
	{
		CStopWatch tmh;
		LIBRARY->initializeLibrary();
		logGlobal->info("Initializing VCMI_Lib: %d ms", tmh.getDiff());
	}
	catch (const DataLoadingException & e)
	{
		criticalInitializationError = e.what();
		return;
	}

	// Debug code to load all maps on start
	//ClientCommandManager commandController;
	//commandController.processCommand("translate maps", false);
}

static void checkForModLoadingFailure()
{
	const auto & brokenMods = LIBRARY->identifiersHandler->getModsWithFailedRequests();
	if (!brokenMods.empty())
	{
		MetaString messageText;
		messageText.appendTextID("vcmi.client.errors.modLoadingFailure");

		for (const auto & modID : brokenMods)
		{
			messageText.appendRawString(LIBRARY->modh->getModInfo(modID).getName());
			messageText.appendEOL();
		}
		CInfoWindow::showInfoDialog(messageText.toString(), {});
	}
}

static void prog_version()
{
	printf("%s\n", GameConstants::VCMI_VERSION.c_str());
	std::cout << VCMIDirs::get().genHelpString();
}

static void prog_help(const po::options_description &opts)
{
	auto time = std::time(nullptr);
	printf("%s - A Heroes of Might and Magic 3 clone\n", GameConstants::VCMI_VERSION.c_str());
	printf("Copyright (C) 2007-%d VCMI dev team - see AUTHORS file\n", std::localtime(&time)->tm_year + 1900);
	printf("This is free software; see the source for copying conditions. There is NO\n");
	printf("warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n");
	printf("\n");
	std::cout << opts;
}

#if defined(VCMI_WINDOWS) && !defined(__GNUC__) && defined(VCMI_WITH_DEBUG_CONSOLE)
int wmain(int argc, wchar_t* argv[])
#elif defined(VCMI_MOBILE)
int SDL_main(int argc, char *argv[])
#else
int main(int argc, char * argv[])
#endif
{
#ifdef VCMI_ANDROID
	CAndroidVMHelper::initClassloader(SDL_AndroidGetJNIEnv());
	// boost will crash without this
	setenv("LANG", "C", 1);
	    std::thread([]() {
        // …
    }).detach();
#endif  // VCMI_ANDROID
#if !defined(VCMI_MOBILE)
	// Correct working dir executable folder (not bundle folder) so we can use executable relative paths
	boost::filesystem::current_path(boost::filesystem::system_complete(argv[0]).parent_path());
#endif
	std::cout << "Starting... " << std::endl;

	// --- SOCKET SERVER THREAD FOR EXTERNAL COMMANDS ---
	std::thread([]() {
		int server_fd, new_socket;
		struct sockaddr_in address;
		int opt = 1;
		int addrlen = sizeof(address);
		char buffer[1024] = {0};

		std::cout << "[SOCKET] Starting socket server thread..." << std::endl;

		if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
			perror("socket failed");
			return;
		}
		std::cout << "[SOCKET] socket() OK" << std::endl;

		if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
			perror("setsockopt");
			return;
		}
		std::cout << "[SOCKET] setsockopt() OK" << std::endl;

		address.sin_family = AF_INET;
		address.sin_addr.s_addr = INADDR_ANY;
		address.sin_port = htons(5000); // port 5000

		if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
			perror("bind failed");
			return;
		}
		std::cout << "[SOCKET] bind() OK" << std::endl;

		if (listen(server_fd, 3) < 0) {
			perror("listen");
			return;
		}
		std::cout << "[SOCKET] listen() OK" << std::endl;

		logGlobal->info("🧠 Socket server listening on port 5000...");
		std::cout << "[SOCKET] Listening on port 5000..." << std::endl;

		while (true)
		{
			if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
				perror("accept");
				continue;
			}

			int valread = read(new_socket, buffer, 1024);
			buffer[valread] = '\0';

			std::string cmd(buffer);
			cmd.erase(std::remove(cmd.begin(), cmd.end(), '\n'), cmd.end());

			logGlobal->info("🛰️ Received command via socket: %s", cmd.c_str());

			if (!GAME)
			{
				logGlobal->warn("🚫 GAME is null.");
				close(new_socket);
				continue;
			}
			else if (cmd == "move_active_hero_left")
			{
				// 1) grab the currently selected (active) hero from the local state
				const CGHeroInstance* h = GAME->interface()->localState->getCurrentHero();
				if (!h)
				{
					logGlobal->warn("No hero currently selected.");
				}
				else
				{
					// 2) compute the target vis tile and convert to world coords
					int3 oldVis = h->visitablePos();
					int3 newVis { oldVis.x - 1, oldVis.y, oldVis.z };
					int3 worldDest = h->convertFromVisitablePos(newVis);

					// 3) send the exact same packet the UI would:
					//    this enqueues a MoveHero pack (path=[worldDest], heroID=h->id, transit=false)
					GAME->interface()->cb->moveHero(h, worldDest, /*useTransit=*/false);
					logGlobal->info("Requested hero move from (%d,%d) to (%d,%d)",
									oldVis.x, oldVis.y, newVis.x, newVis.y);
				}
			}
			// Move the active hero one tile to the right
			else if (cmd == "move_active_hero_right")
			{
				// 1) grab the currently selected hero
				const CGHeroInstance* h = GAME->interface()->localState->getCurrentHero();
				if (!h)
				{
					logGlobal->warn("No hero currently selected.");
					close(new_socket);
					continue;
				}

				// 2) compute current and target visitable‐tile coords
				int3 oldVis = h->visitablePos();
				int3 newVis{ oldVis.x + 1, oldVis.y, oldVis.z };

				// 3) convert that to world coords
				int3 worldDest = h->convertFromVisitablePos(newVis);

				// 4) send the MoveHero packet (false = foot move)
				GAME->interface()->cb->moveHero(h, worldDest, /*useTransit=*/false);

				logGlobal->info("Socket: moved hero from (%d,%d) to (%d,%d)",
								oldVis.x, oldVis.y,
								newVis.x, newVis.y);

				close(new_socket);
				continue;
			}
			else if (cmd == "open_load_menu")
			{
				logGlobal->info("Socket: opening Load Game menu");

				// 2) Call the exact same method your “Load” button invokes:
				CMainMenu::openLobby(
					ESelectionScreen::loadGame,  // the “Load Game” screen
					true,                        // host locally
					{},                          // no player names needed
					ELoadMode::SINGLE            // single‐player load mode
				);

				close(new_socket);
				continue;
			}
			else if (cmd == "lobby_accept")
			{
				logGlobal->info("Socket: simulating Enter on Load-Game");

				// exactly what the Load button/Enter key does:
				if (GAME->server().validateGameStart(false))
					GAME->server().sendStartGame(false);
				else
					logGlobal->warn("Load-Game validation failed");

				close(new_socket);
				continue;
			}
			else
			{
				if (GLOBAL_SOCKET_ACTION_CONTROLLER)
				{
					logGlobal->info("🎯 Sending command to controller directly.");
					GLOBAL_SOCKET_ACTION_CONTROLLER->performSocketCommand(cmd);
				}
				else
				{
					logGlobal->warn("🚫 GLOBAL_SOCKET_ACTION_CONTROLLER is null.");
				}
			}

			close(new_socket);
		}
	}).detach();


	po::options_description opts("Allowed options");
	po::variables_map vm;

	opts.add_options()
		("help,h", "display help and exit")
		("version,v", "display version information and exit")
		("testmap", po::value<std::string>(), "")
		("testsave", po::value<std::string>(), "")
		("logLocation", po::value<std::string>(), "new location for log files")
		("spectate,s", "enable spectator interface for AI-only games")
		("spectate-ignore-hero", "wont follow heroes on adventure map")
		("spectate-hero-speed", po::value<int>(), "hero movement speed on adventure map")
		("spectate-battle-speed", po::value<int>(), "battle animation speed for spectator")
		("spectate-skip-battle", "skip battles in spectator view")
		("spectate-skip-battle-result", "skip battle result window")
		("onlyAI", "allow one to run without human player, all players will be default AI")
		("headless", "runs without GUI, implies --onlyAI")
		("ai", po::value<std::vector<std::string>>(), "AI to be used for the player, can be specified several times for the consecutive players")
		("oneGoodAI", "puts one default AI and the rest will be EmptyAI")
		("autoSkip", "automatically skip turns in GUI")
		("disable-video", "disable video player")
		("nointro,i", "skips intro movies")
		("donotstartserver,d","do not attempt to start server and just connect to it instead server")
		("serverport", po::value<si64>(), "override port specified in config file")
		("savefrequency", po::value<si64>(), "limit auto save creation to each N days");

	if(argc > 1)
	{
		try
		{
			po::store(po::parse_command_line(argc, argv, opts, po_style::unix_style|po_style::case_insensitive), vm);
		}
		catch(boost::program_options::error &e)
		{
			std::cerr << "Failure during parsing command-line options:\n" << e.what() << std::endl;
		}
	}

	po::notify(vm);
	if(vm.count("help"))
	{
		prog_help(opts);
#ifdef VCMI_IOS
		exit(0);
#else
		return 0;
#endif
	}
	if(vm.count("version"))
	{
		prog_version();
#ifdef VCMI_IOS
		exit(0);
#else
		return 0;
#endif
	}

	// Init old logging system and new (temporary) logging system
	CStopWatch total;
	CStopWatch pomtime;
	std::cout.flags(std::ios::unitbuf);

	setThreadNameLoggingOnly("MainGUI");
	boost::filesystem::path logPath = VCMIDirs::get().userLogsPath() / "VCMI_Client_log.txt";
	if(vm.count("logLocation"))
		logPath = vm["logLocation"].as<std::string>() + "/VCMI_Client_log.txt";

#ifndef VCMI_IOS

	auto callbackFunction = [](std::string buffer, bool calledFromIngameConsole)
	{
		ClientCommandManager commandController;
		commandController.processCommand(buffer, calledFromIngameConsole);
	};

	CConsoleHandler console(callbackFunction);
	console.start();

	CBasicLogConfigurator logConfigurator(logPath, &console);
#else
	CBasicLogConfigurator logConfigurator(logPath, nullptr);
#endif

	logConfigurator.configureDefault();
	logGlobal->info("Starting client of '%s'", GameConstants::VCMI_VERSION);
	logGlobal->info("Creating console and configuring logger: %d ms", pomtime.getDiff());
	logGlobal->info("The log file will be saved to %s", logPath);

	// Init filesystem and settings
	try
	{
		LIBRARY = new GameLibrary;
		LIBRARY->initializeFilesystem(false);
	}
	catch (const DataLoadingException & e)
	{
		handleFatalError(e.what(), true);
	}

	Settings session = settings.write["session"];
	auto setSettingBool = [&](const std::string & key, const std::string & arg) {
		Settings s = settings.write(vstd::split(key, "/"));
		if(vm.count(arg))
			s->Bool() = true;
		else if(s->isNull())
			s->Bool() = false;
	};
	auto setSettingInteger = [&](const std::string & key, const std::string & arg, si64 defaultValue) {
		Settings s = settings.write(vstd::split(key, "/"));
		if(vm.count(arg))
			s->Integer() = vm[arg].as<si64>();
		else if(s->isNull())
			s->Integer() = defaultValue;
	};

	setSettingBool("session/onlyai", "onlyAI");
	setSettingBool("session/disableVideo", "disable-video");
	if(vm.count("headless"))
	{
		session["headless"].Bool() = true;
		session["onlyai"].Bool() = true;
	}
	else if(vm.count("spectate"))
	{
		session["spectate"].Bool() = true;
		session["spectate-ignore-hero"].Bool() = vm.count("spectate-ignore-hero");
		session["spectate-skip-battle"].Bool() = vm.count("spectate-skip-battle");
		session["spectate-skip-battle-result"].Bool() = vm.count("spectate-skip-battle-result");
		if(vm.count("spectate-hero-speed"))
			session["spectate-hero-speed"].Integer() = vm["spectate-hero-speed"].as<int>();
		if(vm.count("spectate-battle-speed"))
			session["spectate-battle-speed"].Float() = vm["spectate-battle-speed"].as<int>();
	}
	// Server settings
	setSettingBool("session/donotstartserver", "donotstartserver");

	// Init special testing settings
	setSettingInteger("session/serverport", "serverport", 0);
	setSettingInteger("general/saveFrequency", "savefrequency", 1);

	// Initialize logging based on settings
	logConfigurator.configure();
	logGlobal->debug("settings = %s", settings.toJsonNode().toString());

	// Some basic data validation to produce better error messages in cases of incorrect install
	auto testFile = [](const std::string & filename, const std::string & message)
	{
		if (!CResourceHandler::get()->existsResource(ResourcePath(filename)))
			handleFatalError(message, false);
	};

	testFile("DATA/HELP.TXT", "VCMI requires Heroes III: Shadow of Death or Heroes III: Complete data files to run!");
	testFile("DATA/TENTCOLR.TXT", "Heroes III: Restoration of Erathia (including HD Edition) data files are not supported!");
	testFile("MODS/VCMI/MOD.JSON", "VCMI installation is corrupted!\nBuilt-in mod was not found!");
	testFile("DATA/NOTOSERIF-MEDIUM.TTF", "VCMI installation is corrupted!\nBuilt-in font was not found!\nManually deleting '" + VCMIDirs::get().userDataPath().string() + "/Mods/VCMI' directory (if it exists)\nor clearing app data and reimporting Heroes III files may fix this problem.");
	testFile("DATA/PLAYERS.PAL", "Heroes III data files (Data/H3Bitmap.lod) are incomplete or corruped!\n Please reinstall them.");
	testFile("SPRITES/DEFAULT.DEF", "Heroes III data files (Data/H3Sprite.lod) are incomplete or corruped!\n Please reinstall them.");

	if(!settings["session"]["headless"].Bool())
		ENGINE = std::make_unique<GameEngine>();

	GAME = std::make_unique<GameInstance>();

	if (ENGINE)
		ENGINE->setEngineUser(GAME.get());
	
#ifndef VCMI_NO_THREADED_LOAD
	//we can properly play intro only in the main thread, so we have to move loading to the separate thread
	std::thread loading([]()
	{
		setThreadName("initialize");
		init();
	});
#else
	init();
#endif

#ifndef VCMI_NO_THREADED_LOAD
	#ifdef VCMI_ANDROID // android loads the data quite slowly so we display native progressbar to prevent having only black screen for few seconds
	{
		CAndroidVMHelper vmHelper;
		vmHelper.callStaticVoidMethod(CAndroidVMHelper::NATIVE_METHODS_DEFAULT_CLASS, "showProgress");
	#endif // ANDROID
		loading.join();
	#ifdef VCMI_ANDROID
		vmHelper.callStaticVoidMethod(CAndroidVMHelper::NATIVE_METHODS_DEFAULT_CLASS, "hideProgress");
	}
	#endif // ANDROID
#endif // THREADED

	if (criticalInitializationError.has_value())
	{
		handleFatalError(criticalInitializationError.value(), false);
	}

	if (ENGINE)
	{
		pomtime.getDiff();
		graphics = new Graphics(); // should be before curh
		ENGINE->renderHandler().onLibraryLoadingFinished(LIBRARY);

		CMessage::init();
		logGlobal->info("Message handler: %d ms", pomtime.getDiff());

		ENGINE->cursor().init();
		ENGINE->cursor().show();
	}

	logGlobal->info("Initialization of VCMI (together): %d ms", total.getDiff());

	session["autoSkip"].Bool()  = vm.count("autoSkip");
	session["oneGoodAI"].Bool() = vm.count("oneGoodAI");
	session["aiSolo"].Bool() = false;
	
	if(vm.count("testmap"))
	{
		session["testmap"].String() = vm["testmap"].as<std::string>();
		session["onlyai"].Bool() = true;
		GAME->server().debugStartTest(session["testmap"].String(), false);
	}
	else if(vm.count("testsave"))
	{
		session["testsave"].String() = vm["testsave"].as<std::string>();
		session["onlyai"].Bool() = true;
		GAME->server().debugStartTest(session["testsave"].String(), true);
	}
	else if (!settings["session"]["headless"].Bool())
	{
		GAME->mainmenu()->makeActiveInterface();

		bool playIntroVideo = !vm.count("battle") && !vm.count("nointro") && settings["video"]["showIntro"].Bool();
		if(playIntroVideo)
			GAME->mainmenu()->playIntroVideos();
		else
			GAME->mainmenu()->playMusic();
	}
	
#ifndef VCMI_UNIX
	// on Linux, name of main thread is also name of our process. Which we don't want to change
	setThreadName("MainGUI");
#endif

	try
	{
		if (ENGINE)
		{
			checkForModLoadingFailure();
			ENGINE->mainLoop();
		}
		else
		{
			while(!headlessQuit)
				std::this_thread::sleep_for(std::chrono::milliseconds(200));

			std::this_thread::sleep_for(std::chrono::milliseconds(500));
		}
	}
	catch (const GameShutdownException & )
	{
		// no-op - just break out of main loop
		logGlobal->info("Main loop termination requested");
	}

	GAME->server().endNetwork();

	if(!settings["session"]["headless"].Bool())
	{
		if(GAME->server().client)
			GAME->server().endGameplay();

		if (ENGINE)
			ENGINE->windows().clear();
	}

	GAME.reset();

	if(!settings["session"]["headless"].Bool())
	{
		CMessage::dispose();
		delete graphics;
		graphics = nullptr;
	}

	// must be executed before reset - since unique_ptr resets pointer to null before calling destructor
	ENGINE->async().wait();

	ENGINE.reset();

	delete LIBRARY;
	LIBRARY = nullptr;
	logConfigurator.deconfigure();

	std::cout << "Ending...\n";
	return 0;
}

/// Notify user about encountered fatal error and terminate the game
/// TODO: decide on better location for this method
void handleFatalError(const std::string & message, bool terminate)
{
	logGlobal->error("FATAL ERROR ENCOUNTERED, VCMI WILL NOW TERMINATE");
	logGlobal->error("Reason: %s", message);

	std::string messageToShow = "Fatal error! " + message;

	SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Fatal error!", messageToShow.c_str(), nullptr);

	if (terminate)
		throw std::runtime_error(message);
	else
		::exit(1);
}
