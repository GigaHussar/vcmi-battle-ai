/*
 * mapsettings.h, part of VCMI engine
 *
 * Authors: listed in file AUTHORS in main folder
 *
 * License: GNU General Public License v2.0 or later
 * Full text of license available in license.txt file, in main folder
 *
 */

#pragma once

#include <QDialog>
#include "mapcontroller.h"
#include "../lib/mapping/CMap.h"

namespace Ui {
class MapSettings;
}

class ModSettings;

class MapSettings : public QDialog
{
	Q_OBJECT

public:
	explicit MapSettings(MapController & controller, QWidget *parent = nullptr);
	~MapSettings();

	ModSettings * getModSettings();

private slots:
	void on_pushButton_clicked();

private:
	
	Ui::MapSettings *ui;
	MapController & controller;
};
