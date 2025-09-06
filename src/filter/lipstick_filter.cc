/*
 * GPUPixel
 *
 * Created by PixPark on 2021/6/24.
 * Copyright © 2021 PixPark. All rights reserved.
 */

#include "gpupixel/filter/lipstick_filter.h"
#include "core/gpupixel_context.h"
#include "gpupixel/source/source_image.h"
#include "utils/util.h"
namespace gpupixel {

LipstickFilter::LipstickFilter() {}

std::shared_ptr<LipstickFilter> LipstickFilter::Create() {
  auto ret = std::shared_ptr<LipstickFilter>(new LipstickFilter());
  gpupixel::GPUPixelContext::GetInstance()->SyncRunWithContext([&] {
    if (ret && !ret->Init()) {
      ret.reset();
    }
  });
  return ret;
}

bool LipstickFilter::Init() {
  auto path = Util::GetResourcePath() / "res";
  auto mouth = SourceImage::Create((path / "mouth.png").string());
  SetImageTexture(mouth);
  // Adjust bounds for 640x480 camera resolution
  SetTextureBounds(FrameBounds{160, 320, 320, 160});  // Center region for testing
  return FaceMakeupFilter::Init();
}

}  // namespace gpupixel
