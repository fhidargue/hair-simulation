#!/usr/bin/env -S uv run --script

import sys

from ncca.ngl.vec3 import Vec3
from ncca.ngl.widgets import RGBColourWidget
from PySide6.QtCore import QFile
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFrame, QMainWindow, QVBoxLayout, QWidget

from lib.constants import HAIR_COLOR, HAIR_STYLES
from ui.debug_app import DebugApplication
from ui.hair_scene import HairScene


class Loader(QUiLoader):
    def createWidget(self, class_name, parent=None, name=""):
        if class_name == "RGBColourWidget":
            return RGBColourWidget(parent)
        return super().createWidget(class_name, parent, name)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.load_ui("ui/qt/hair_widget.ui")
        self.resize(1200, 720)
        self.scene = HairScene()
        self._set_custom_widgets()
        self._connect_signals_and_slots()
        self._set_scene_widget_layout()

    def _set_scene_widget_layout(self):
        """
        Creates and applies the main Qt layout that embeds the HairScene OpenGL widget into the UI.
        """
        layout = QVBoxLayout(self.scene_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scene)

    def _set_custom_widgets(self):
        """
        Creates and configures all custom Qt control widgets (sliders, buttons, checkboxes)
        used to interact with the hair simulation and rendering parameters.
        """
        self.hair_color.set_colour(Vec3(*HAIR_COLOR))
        self.hair_color.setFrameShape(QFrame.Shape.NoFrame)
        self.hair_style.addItems(HAIR_STYLES)
        self.hair_style.setCurrentIndex(0)

    def _connect_signals_and_slots(self):
        """
        Connects Qt widget signals (sliders, buttons, checkboxes) to their corresponding
        scene update slots so UI changes immediately affect the hair simulation and rendering.
        """
        self.hair_strand_count.valueChanged.connect(self.scene.update_hair_strand_count)
        self.hair_color.colourChanged.connect(self.scene.update_hair_color)
        self.hair_style.activated.connect(self.scene.update_hair_style)
        self.hair_length.valueChanged.connect(self.scene.update_hair_length)
        self.hair_length.sliderReleased.connect(self.scene._rebuild_hair)
        self.hair_thickness.valueChanged.connect(self.scene.update_hair_thickness)
        self.hair_density.valueChanged.connect(self.scene.update_hair_density)
        self.activate_wind.stateChanged.connect(self.scene.update_wind)
        self.show_ellipsoids.stateChanged.connect(self.scene.update_show_ellipsoids)

    def load_ui(self, ui_file_name: str) -> None:
        """
        Load a .ui file and set up the widgets as attributes of this class.
        """
        try:
            loader = Loader()
            ui_file = QFile(ui_file_name)
            ui_file.open(QFile.OpenModeFlag.ReadOnly)
            loaded_ui = loader.load(ui_file, self)
            self.setCentralWidget(loaded_ui)

            for child in loaded_ui.findChildren(QWidget):
                name = child.objectName()

                if name:
                    setattr(self, name, child)

            ui_file.close()
        except Exception:
            print(f"There was an issue loading the Qt UI file {ui_file_name}")
            raise


def main():
    fmt = QSurfaceFormat()
    fmt.setMajorVersion(4)
    fmt.setMinorVersion(1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    if len(sys.argv) > 1 and "--debug" in sys.argv:
        app = DebugApplication(sys.argv)
    else:
        app = QApplication(sys.argv)

    win = MainWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
