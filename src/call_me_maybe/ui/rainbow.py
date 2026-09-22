from colorsys import hsv_to_rgb

from rich.highlighter import Highlighter


class RainbowHighlighter(Highlighter):
    def highlight(self, text: str) -> None:
        length = max(len(text) - 1, 1)

        for index in range(len(text)):
            hue = index / length
            r, g, b = hsv_to_rgb(hue, 1.0, 1.0)

            color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            text.stylize(color, index, index + 1)
