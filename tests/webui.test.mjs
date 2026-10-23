import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../python/omni_tokenizer/web/app.js", import.meta.url));
const {resizeLimits, initialSize, resizeSize, fullCrop, cropBounds, dragCrop, resizeTo, formatTokenText} = await import(`data:text/javascript;base64,${source.toString("base64")}`);

test("resize defaults and explicit limits support dimensions above 1152", () => {
  assert.deepEqual(initialSize(2400, 1600), [1152, 768]);
  assert.deepEqual(initialSize(2400, 1600, 512, 2048), [2048, 1365]);
  assert.deepEqual(initialSize(100, 150, 512, 2048), [512, 768]);
  assert.deepEqual(initialSize(192, 288, 512, 96), [64, 96]);
  assert.deepEqual(resizeTo(900, 600, "width", 1920, true, 2048), [1920, 1280]);
  assert.deepEqual(fullCrop(...initialSize(2400, 1600, 512, 2048)), {x: 0, y: 2, w: 2048, h: 1360});
  for (const invalid of [true, 0, 17, NaN, "2048", 2 ** 32]) assert.throws(() => resizeLimits(384, invalid));
});

test("newline escaping changes only row separators and can be switched off", () => {
  const raw = "<|image_7|><|image_8|>\n<|image_9|><|image_10|>";
  assert.equal(formatTokenText(raw, true), "<|image_7|><|image_8|>\\n<|image_9|><|image_10|>");
  assert.equal(formatTokenText(raw, false), raw);
  assert.equal(formatTokenText("<|audio_17|><|audio_230|>", true), "<|audio_17|><|audio_230|>");
  assert.equal(formatTokenText("a\r\nb\rc", true), "a\\nb\\nc");
});

test("resize preserves the aspect ratio; alignment removes pixels instead of stretching", () => {
  assert.deepEqual(resizeTo(900, 600, "width", 800, true), [800, 533]);
  assert.deepEqual(fullCrop(800, 533), {x: 0, y: 2, w: 800, h: 528});
  assert.deepEqual(fullCrop(900, 600), {x: 2, y: 4, w: 896, h: 592});
  assert.deepEqual(resizeTo(600, 400, "height", 512, false), [600, 512]);
  assert.deepEqual(resizeTo(600, 400, "width", 60000, true), [1152, 768]);
  assert.throws(() => resizeTo(600, 400, "width", NaN, true));
});

test("crop selections snap to whole patches and stay within resized pixels", () => {
  assert.deepEqual(dragCrop({x: 90, y: 70}, {x: 10, y: 20}, {}, "new", 100, 80), {x: 10, y: 22, w: 80, h: 48});
  assert.deepEqual(dragCrop({x: 0, y: 0}, {x: 200, y: -100}, {x: 10, y: 20, w: 48, h: 32}, "move", 100, 80), {x: 52, y: 0, w: 48, h: 32});
  assert.deepEqual(cropBounds({x: -10, y: 80, w: 200, h: 0}, 100, 80), {x: 0, y: 64, w: 96, h: 16});
});

test("corner resize keeps the opposite corner and supports crossing it", () => {
  const rect = {x: 10, y: 20, w: 48, h: 32};
  assert.deepEqual(dragCrop({}, {x: 26, y: 20}, rect, "nw", 100, 80), {x: 26, y: 20, w: 32, h: 32});
  assert.deepEqual(dragCrop({}, {x: 0, y: 4}, rect, "se", 100, 80), {x: 0, y: 4, w: 16, h: 16});
});

test("alignment always crops within the resized image and needs no extra scaling", () => {
  for (const width of [16, 31, 100, 384, 392, 600, 1152]) {
    for (const height of [16, 29, 408, 900, 1152]) {
      for (const edge of [16, 384, 576, 900, 1152, 8000]) {
        const [w, h] = resizeSize(width, height, edge);
        assert.ok(w >= 16 && h >= 16 && w <= 1152 && h <= 1152);
        const rect = fullCrop(w, h);
        assert.ok(rect.w % 16 === 0 && rect.h % 16 === 0);
        assert.ok(rect.x >= 0 && rect.y >= 0 && rect.x + rect.w <= w && rect.y + rect.h <= h);
        assert.ok(w - rect.w < 16 && h - rect.h < 16);
      }
    }
  }
});
