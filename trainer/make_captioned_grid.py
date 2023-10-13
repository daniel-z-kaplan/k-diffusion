from torch import FloatTensor
from PIL import Image, ImageFont, ImageDraw
from typing import List, NamedTuple, Protocol
from dataclasses import dataclass
from textwrap import TextWrapper
from functools import partial

@dataclass
class FontMetrics:
  chartop: int
  charleft: int
  charw: int
  charh: int
  line_spacing: int

class BBox(NamedTuple):
  top: int
  left: int
  bottom: int
  right: int

# fnt = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSansMono.ttf", 30)

def get_font_metrics(font: ImageFont):
  tmp = Image.new("RGB", (100, 100))
  draw = ImageDraw.Draw(tmp)
  bbox = draw.textbbox((0, 0), "M", font=font)
  left, top, right, bottom = bbox
  charw = right-left
  charh = bottom-top

  bbox2 = draw.textbbox((0, 0), "M\nM", font=font)
  _, top_, _, bottom_ = bbox2
  line_spacing = (bottom_-top_)-2*charh

  return FontMetrics(
    chartop=top,
    charleft=left,
    charw=charw,
    charh=charh,
    line_spacing=line_spacing,
  )

# textw = 150
# textwr = TextWrapper(width=textw//font_metrics.charw)

def make_captioned_grid(
  wrapper: TextWrapper,
  font: ImageFont,
  font_metrics: FontMetrics,
  margins: BBox,
  img: FloatTensor,
  captions: List[List[str]],
) -> Image.Image:
  """
  Args:
    img (`FloatTensor`) a [... height width channels] tensor. zero-centered, RGB, tolerates being clamped to ±1 and moved to CPU
  Return:
    PIL `Image`; a grid, with captions
  """
  textw = 150
  # textwr = TextWrapper(width=textw//font_metrics.charw)

  lines = wrapper.wrap('hard-leaved pocket orchid')
  wrapped = "\n".join(lines)
  print(wrapped)

  line_count = len(lines)
  texth = line_count*font_metrics.charh+(line_count-1)* font_metrics.line_spacing

  margin_top, margin_left, margin_bottom, margin_right = margins

  out = Image.new("RGB", (textw+margin_left+margin_right, texth+margin_top+margin_bottom), (255, 255, 255))
  d = ImageDraw.Draw(out)

  d.multiline_text((margin_left-font_metrics.charleft, margin_top-font_metrics.chartop), wrapped, font=font, fill=(0, 0, 0))

  return out

class GridCaptioner(Protocol):
  @classmethod
  def __call__(img: FloatTensor, captions: List[List[str]]) -> Image.Image: ...

def make_grid_captioner(
  wrapper: TextWrapper,
  font: ImageFont,
  font_metrics: FontMetrics,
  margins: BBox
) -> GridCaptioner:
  return partial(
    make_captioned_grid,
    wrapper=wrapper,
    font=font,
    font_metrics=font_metrics,
    margins=margins,
  )