from PIL import Image, ImageFont, ImageDraw
from PIL.ImageFont import FreeTypeFont
from typing import List, NamedTuple, Protocol, Optional
from dataclasses import dataclass
from textwrap import TextWrapper
from functools import partial
import math
import numpy as np
from numpy.typing import NDArray

from .iteration.batched import batched

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

def make_captioned_grid(
  wrapper: TextWrapper,
  font: FreeTypeFont,
  font_metrics: FontMetrics,
  padding: BBox,
  cols: int,
  samp_w: int,
  samp_h: int,
  imgs: List[Image.Image],
  captions: List[str],
) -> Image.Image:
  """
  Args:
    font `FreeTypeFont` for example: ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSansMono.ttf", 30)
    imgs `List[Image.Image]` images (which we will tabulate into rows & columns)
  Return:
    PIL `Image`; a grid, with captions
  """
  assert len(imgs) == len(captions)
  assert len(imgs) > 0

  pad_top, pad_left, pad_bottom, pad_right = padding
  # textw: int = samp_w-(pad_left+pad_right)

  rows: int = math.ceil(len(imgs)/cols)

  # compute all wrapped captions first, so that we can know max line counts in advance of allocating canvas
  wrappeds: List[List[str]] = []
  text_heights: List[int] = []
  for captions_ in batched(captions, cols):
    lines: List[List[str]] = [wrapper.wrap(caption) for caption in captions_]
    line_counts: List[int] = [len(lines_) for lines_ in lines]
    max_line_count: int = max(line_counts)
    text_height: int = max_line_count*font_metrics.charh+(max_line_count-1)* font_metrics.line_spacing
    text_heights.append(text_height)

    wrappeds_: List[str] = ["\n".join(lines_) for lines_ in lines]
    wrappeds.append(wrappeds_)

  text_heights_np: NDArray = np.array(text_heights)
  text_heights_cumsum = np.roll(text_heights_np.cumsum(), 1)
  text_heights_cumsum[0] = 0

  out = Image.new("RGB", (samp_w*cols, text_heights_np.sum()+rows*(pad_top+pad_bottom+samp_h)), (255, 255, 255))
  text_x_offset: int = pad_left - font_metrics.charleft
  text_y_offset: int = pad_top - font_metrics.chartop
  d = ImageDraw.Draw(out)
  for row_ix, (imgs_, wrappeds_, text_heights_cumsum_, current_text_height) in enumerate(zip(batched(imgs, cols), wrappeds, text_heights_cumsum, text_heights_np)):
    row_y: int = text_heights_cumsum_ + row_ix * (pad_top + pad_bottom + samp_h)
    text_y: int = row_y + text_y_offset
    img_y: int = row_y + pad_top + current_text_height + pad_bottom
    for col_ix, (img, wrapped) in enumerate(zip(imgs_, wrappeds_)):
      col_x: int = col_ix * samp_w
      text_x: int = col_x + text_x_offset
      d.multiline_text((text_x, text_y), wrapped, font=font, fill=(0, 0, 0))
      out.paste(img, box=(col_x, img_y))

  return out

class GridCaptioner(Protocol):
  @classmethod
  def __call__(
    imgs: List[Image.Image],
    captions: List[str],
  ) -> Image.Image: ...

class TextWrapperFactory(Protocol):
  @classmethod
  def __call__(
    width: int,
  ) -> TextWrapper: ...

def make_grid_captioner(
  font: FreeTypeFont,
  font_metrics: FontMetrics,
  padding: BBox,
  cols: int,
  samp_w: int,
  samp_h: int,
  wrapper_factory: Optional[TextWrapper] = TextWrapper,
) -> GridCaptioner:
  textw = samp_w - (padding.left + padding.right)
  wrap_at = textw//font_metrics.charw
  textwr: TextWrapper = wrapper_factory(width=wrap_at)
  return partial(
    make_captioned_grid,
    wrapper=textwr,
    font=font,
    font_metrics=font_metrics,
    padding=padding,
    cols=cols,
    samp_w=samp_w,
    samp_h=samp_h,
  )