/*
 * ScriptFixture.cpp, part of VCMI engine
 *
 * Authors: listed in file AUTHORS in main folder
 *
 * License: GNU General Public License v2.0 or later
 * Full text of license available in license.txt file, in main folder
 *
 */
#include "StdInc.h"
#include "lib/modding/ModScope.h"
#include "lib/GameLibrary.h"

#include "ScriptFixture.h"

namespace test
{
using namespace ::testing;
using namespace ::scripting;

ScriptFixture::ScriptFixture() =  default;

ScriptFixture::~ScriptFixture() = default;

void ScriptFixture::loadScriptFromFile(const std::string & path)
{
	JsonNode scriptConfig;
	scriptConfig["source"].String() = path;
	loadScript(scriptConfig);
}

void ScriptFixture::loadScript(const JsonNode & scriptConfig)
{
	subject = LIBRARY->scriptHandler->loadFromJson(&loggerMock, ModScope::scopeBuiltin(), scriptConfig, "test");

	GTEST_ASSERT_NE(subject, nullptr);

	context = subject->createContext(&environmentMock);

	EXPECT_CALL(*pool, getContext(Eq(subject.get()))).WillRepeatedly(Return(context));
}

void ScriptFixture::loadScript(ModulePtr modulePtr, const std::string & scriptSource)
{
	subject = std::make_shared<ScriptImpl>(LIBRARY->scriptHandler.get());

	subject->host = modulePtr;
	subject->sourceText = scriptSource;
	subject->identifier = "test";
	subject->compile(&loggerMock);

	context = subject->createContext(&environmentMock);

	EXPECT_CALL(*pool, getContext(Eq(subject.get()))).WillRepeatedly(Return(context));
}

void ScriptFixture::loadScript(ModulePtr modulePtr, const std::vector<std::string> & scriptSource)
{
	std::string source = boost::algorithm::join(scriptSource, "\n");

	loadScript(modulePtr, source);
}

void ScriptFixture::setUp()
{
	pool = std::make_shared<PoolMock>();

	EXPECT_CALL(environmentMock, game()).WillRepeatedly(Return(&infoMock));
	EXPECT_CALL(environmentMock, logger()).WillRepeatedly(Return(&loggerMock));
	EXPECT_CALL(environmentMock, eventBus()).WillRepeatedly(Return(&eventBus));
	EXPECT_CALL(environmentMock, services()).WillRepeatedly(Return(&servicesMock));
}

JsonNode ScriptFixture::runClientServer(const JsonNode & scriptState)
{
	context->run(scriptState);
	return context->saveState();
}

JsonNode ScriptFixture::runServer(const JsonNode & scriptState)
{
	context->run(&serverMock, scriptState);
	return context->saveState();
}

JsonNode ScriptFixture::runScript(ModulePtr modulePtr, const std::string & scriptSource, const JsonNode & scriptState)
{
	loadScript(modulePtr, scriptSource);
	return runClientServer(scriptState);
}


}
