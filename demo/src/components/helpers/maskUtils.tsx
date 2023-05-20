// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// Convert the onnx model mask prediction to ImageData
function arrayToImageData(input: any, width: number, height: number) {
  // Speed up mask conversion by writing uint32's instead of 4 uint8's.
  // This only works on logical endian machines, but realistically,
  // all the machines of interest currently are.
  // const [r, g, b, a] = [0, 114, 189, 255]; // the masks's blue color
  const maskValue = (0 << 0) | (114 << 8) | (189 << 16) | (255 << 24);

  const arr = new Uint8ClampedArray(4 * width * height); // No need to fill - already zeroed.
  const arr32 = new Uint32Array(arr.buffer);

  for (let i = 0; i < input.length; i++) {
    // Threshold the onnx model mask prediction at 0.0
    // This is equivalent to thresholding the mask using predictor.model.mask_threshold
    // in Python.
    if (input[i] > 0.0) {
      arr32[i] = maskValue;
      // arr[4 * i + 0] = r;
      // arr[4 * i + 1] = g;
      // arr[4 * i + 2] = b;
      // arr[4 * i + 3] = a;
    }
  }
  return new ImageData(arr, height, width);
}

// Use a Canvas element to produce an image from ImageData
function imageDataToImage(imageData: ImageData) {
  const canvas = imageDataToCanvas(imageData);
  const image = new Image();
  image.src = canvas.toDataURL();
  return image;
}

// Canvas elements can be created from ImageData
function imageDataToCanvas(imageData: ImageData) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  ctx?.putImageData(imageData, 0, 0);
  return canvas;
}

// Convert the onnx model mask output to an HTMLImageElement
export function onnxMaskToImage(input: any, width: number, height: number) {
  return imageDataToImage(arrayToImageData(input, width, height));
}
