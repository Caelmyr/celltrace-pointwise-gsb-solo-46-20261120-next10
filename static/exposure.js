import {
  $, button, definition, e, field, format, metrics, num, percent, readingLabel, request, submit, wireForm,
} from "./ui.js";

const DEFAULT_MERGE = {
  enabled: false, short_id: null, short_calibration: {}, readings: "composite",
  agreement: 0.15, minimum_agreement_pixels: 30,
};

export function showExposureMerge(app) {
  const merge = app.case.merge || DEFAULT_MERGE;
  const highs = Object.values(app.case.captures).filter((frame) => frame.role === "high");
  const long = app.case.captures[app.case.active.high];
  const others = highs.filter((frame) => frame.id !== app.case.active.high);
  const darkOptions = [["", "Not set"], ...Object.values(app.case.captures)
    .filter((frame) => frame.role === "dark")
    .map((frame) => [frame.id, `${frame.name} · ${frame.exposure_ms} ms`])];
  const flatOptions = [["", "Not set"], ...Object.values(app.case.captures)
    .filter((frame) => frame.role === "flat")
    .map((frame) => [frame.id, `${frame.name} · ${frame.exposure_ms} ms`])];
  const calibration = merge.short_calibration || {};
  const root = app.openDialog("High-current exposures", `<form class="section-stack">
    <p class="small-copy">The selected high-current image stays the long exposure with its own corners and calibration.
    Pair a shorter exposure so its overexposed patch can still be measured. Each exposure keeps its own corners,
    dark reference and display level; the composed plane converts the short exposure by exposure ratio before
    deciding what can be cross-checked.</p>
    <div class="fields">
      ${field("Long exposure (selected)", "long", long?.name, {disabled: true, wide: true})}
      ${others.length
        ? field("Short high-current exposure", "short_id", merge.short_id,
          {values: [["", "Not paired"], ...others.map((frame) =>
            [frame.id, `${frame.name} · ${frame.exposure_ms} ms · ${frame.current_a ?? "—"} A`])], wide: true})
        : '<p class="warning wide">Add a second high-current original with a shorter exposure in Captures first.</p>'}
      ${others.length ? field("Short-exposure dark reference", "short_dark", calibration.dark_id || "", {values: darkOptions, wide: true}) : ""}
      ${others.length ? field("Short-exposure flat reference", "short_flat", calibration.flat_id || "", {values: flatOptions, wide: true}) : ""}
      ${num("Agreement tolerance (fraction)", "agreement", merge.agreement ?? 0.15, .01, .6, .01)}
      ${num("Minimum cross-check pixels", "minimum_agreement_pixels", merge.minimum_agreement_pixels ?? 30, 4, 4000)}
    </div>
    <p class="form-error" hidden></p><div class="form-actions">
      ${merge.enabled ? button("merge-off", "Use one image again", "danger") : ""}
      <button type="submit" value="preview" formnovalidate>Preview split</button>
      ${submit(merge.enabled ? "Save exposure pair" : "Enable exposure merge", "save")}</div></form>
    <div id="merge-preview" class="section-stack"></div>`, true);
  const payloadFor = (values, enabled) => ({
    enabled, short_id: values.short_id,
    short_calibration: {dark_id: values.short_dark || null, flat_id: values.short_flat || null},
    agreement: values.agreement, minimum_agreement_pixels: values.minimum_agreement_pixels,
    readings: merge.readings || "composite",
  });
  wireForm(root, async (values, _, command) => {
    if (command.value === "preview") {
      $("#merge-preview", root).innerHTML = '<p class="small-copy">Composing…</p>';
      try {
        const probe = await request(app.url("/merge/preview"), "POST",
          {...payloadFor(values, true), revision: app.case.revision});
        $("#merge-preview", root).innerHTML = probe.summary
          ? mergePreview(app, probe)
          : '<p class="small-copy">Choose both exposures, then save to compose.</p>';
      } catch (error) {
        $("#merge-preview", root).innerHTML = `<p class="error">${e(error.message)}</p>`;
      }
      return;
    }
    await app.patch("merge", payloadFor(values, true), "High-current exposure reading saved.");
    app.modal.close();
  });
  $("#merge-off", root)?.addEventListener("click", async (event) => {
    event.preventDefault();
    try {
      await app.patch("merge", {
        enabled: false, short_id: merge.short_id, short_calibration: merge.short_calibration,
        agreement: merge.agreement, minimum_agreement_pixels: merge.minimum_agreement_pixels,
        readings: "composite",
      }, "High-current reading uses the single selected image again.");
      app.modal.close();
    } catch (error) {
      app.fail(error);
    }
  });
  if (merge.enabled) {
    $("#merge-preview", root).innerHTML = '<p class="small-copy">Reading the current split…</p>';
    request(app.url("/merge")).then((probe) => {
      if (root.isConnected && probe.summary) $("#merge-preview", root).innerHTML = mergePreview(app, probe);
    }).catch(() => {});
  }
}

function mergePreview(app, probe) {
  const summary = probe.summary;
  return `${metrics([
    ["Corroborated", percent(summary.corroborated_pixels / Math.max(1, summary.pixels))],
    ["Long only", percent(summary.long_only_pixels / Math.max(1, summary.pixels))],
    ["Short only", percent(summary.short_only_pixels / Math.max(1, summary.pixels))],
    ["No conclusion", percent(summary.undecided_pixels / Math.max(1, summary.pixels))],
  ])}<img class="pair-image" src="${e(app.url("/layer/merge.png", {role: "high", grid: 1}))}"
     alt="Exposure merge support partition">
  ${(summary.messages || []).map((message) => `<p class="warning">${e(message)}</p>`).join("")}`;
}

export function showSingleReading(app) {
  const root = app.openDialog("Read the single image", `<div class="section-stack">
    <p class="small-copy">This is the same reading the long exposure gives on its own — its own corners, dark
    reference and raw values. Use it to confirm a composed value has not shifted.</p>
    <div id="single-reading"><p class="small-copy">Select a cell in the image to compare.</p></div>
    <div class="toolbar">${button("single-layer", "Open single-image view", "", app.case.state === "issued")}</div></div>`, true);
  $("#single-layer", root).onclick = async () => {
    try {
      await app.patch("merge", {
        enabled: true, short_id: app.case.merge.short_id, readings: "long",
        short_calibration: app.case.merge.short_calibration,
        agreement: app.case.merge.agreement, minimum_agreement_pixels: app.case.merge.minimum_agreement_pixels,
      }, "High-current reading uses the long exposure only.");
      app.modal.close();
      app.showTab("bench");
    } catch (error) {
      app.fail(error);
    }
  };
  if (app.workspace.cell) showCellReading(app, $("#single-reading", root), app.workspace.cell);
}

export function showCellReadingDialog(app, cell) {
  const root = app.openDialog(`Single-image reading · ${cell}`, `<div class="section-stack">
    <p class="small-copy">The composed value and the long exposure read on its own, for the same cell.</p>
    <div id="single-reading"></div></div>`, true);
  showCellReading(app, $("#single-reading", root), cell);
}

async function showCellReading(app, target, cell) {
  try {
    const value = await request(app.url(`/merge/cell/${cell}`));
    if (!target.isConnected) return;
    target.innerHTML = `${definition([
      ["Cell", cell],
      ["Active reading", value.merge ? readingLabel(value.reading) : "—"],
      ["Merged median", format(value.merge?.median)],
      ["Merged support", value.merge ? percent(value.merge.support) : "—"],
      ["Short only", value.merge ? percent(value.merge.short_only) : "—"],
      ["Long-exposure median", format(value.single?.intensity?.median)],
      ["Long-exposure support", value.single ? percent(value.single.intensity.support) : "—"],
    ])}<p class="small-copy">The long-exposure column is the same reading the long image gives on its own.
      When a single-exposure mode is active, the active value above is the raw short or long exposure.</p>
    ${value.single?.error ? `<p class="warning">${e(value.single.error)}</p>` : ""}`;
  } catch (error) {
    if (target.isConnected) target.innerHTML = `<p class="error">${e(error.message)}</p>`;
  }
}
