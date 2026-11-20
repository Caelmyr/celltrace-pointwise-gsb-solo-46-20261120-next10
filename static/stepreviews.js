import {
  $, $$, ROLE, badge, bindCanvas, button, changeLabel, dateLabel, definition, e, field, format,
  itemCard, metrics, normalizedPoint, percent, request, submit, table, wireForm,
} from "./ui.js";

const VERDICT_LABEL = {
  new: "New after step", grown: "Grown", shrunk: "Shrunk", steady: "Steady",
  merged: "Merged", split: "Split", regrouped: "Regrouped", cleared: "Cleared",
};
const VERDICT_BADGE = {
  new: "alert", grown: "training", merged: "training", split: "training",
  regrouped: "training", cleared: "", steady: "neutral",
};

const view = {
  caseId: null,
  reviewId: null,
  sel: {before: new Set(), after: new Set()},
  redraw: {before: null, after: null},
};

export function renderStepReviews(app, root) {
  if (view.caseId !== app.case.id) {
    view.caseId = app.case.id;
    view.reviewId = null;
    view.sel = {before: new Set(), after: new Set()};
  }
  const reviews = app.case.reviews || [];
  const current = reviews.find((item) => item.id === view.reviewId);
  if (view.reviewId && !current) view.reviewId = null;
  if (view.reviewId) renderDetail(app, root, view.reviewId);
  else renderList(app, root, reviews);
}

function renderList(app, root, reviews) {
  const issued = app.case.state === "issued";
  const frames = Object.values(app.case.captures).filter((frame) => ["high", "low"].includes(frame.role));
  const cards = reviews.map((review) => `
    <article class="card"><div class="card-pad section-stack">
      <div class="toolbar">${badge(changeLabel(review.status), review.status === "current" ? "" : "training")}
        ${badge(`${review.links} confirmed group${review.links === 1 ? "" : "s"}`, "neutral")}</div>
      <h3>${e(review.title)}</h3>
      <p class="small-copy">${e(review.before_step)} · ${e(review.before_name || "missing original")}<br>
        → ${e(review.after_step)} · ${e(review.after_name || "missing original")}</p>
      <p class="small-copy">Updated ${dateLabel(review.updated_at)}</p>
      <div class="row-actions">
        <button type="button" class="compact" data-open="${review.id}">Open review</button>
        ${issued ? "" : `<button type="button" class="compact danger" data-remove="${review.id}">Delete</button>`}
      </div>
    </div></article>`).join("");
  root.innerHTML = `<div class="page-stack">
    ${itemCard("Process step reviews", `<div class="toolbar">
        ${button("new-review", "New step review", "primary", issued || frames.length < 2)}</div>
      <p class="small-copy">Compare dark regions between two process steps of the same module — for example the
        high-current checks before and after lamination. Place the four corners for each original first; the review
        then pins both captures, their placements and the current thresholds so the saved grouping stays traceable.</p>
      ${frames.length < 2 ? '<p class="warning">A step review needs at least two EL captures. Upload the second high-current original under Captures.</p>' : ""}`)}
    ${reviews.length ? `<div class="capture-grid">${cards}</div>` : '<p class="empty-line">No step reviews saved yet.</p>'}
  </div>`;
  $("#new-review", root).onclick = () => createDialog(app, frames);
  $$("[data-open]", root).forEach((node) => node.onclick = () => {
    view.reviewId = node.dataset.open;
    view.sel = {before: new Set(), after: new Set()};
    app.render();
  });
  $$("[data-remove]", root).forEach((node) => node.onclick = () =>
    removeDialog(app, reviews.find((item) => item.id === node.dataset.remove)));
}

function createDialog(app, frames) {
  const ordered = [...frames].sort((a, b) => a.captured_at.localeCompare(b.captured_at) || a.id.localeCompare(b.id));
  const options = ordered.map((frame) => [
    frame.id,
    `${frame.name} · ${ROLE[frame.role]}${frame.current_a != null ? ` · ${frame.current_a} A` : ""}`,
  ]);
  const root = app.openDialog("New step review", `<form><div class="fields">
    ${field("Review title", "title", "Lamination step review", {required: true, wide: true, max: 120})}
    ${field("Earlier step", "before_step", "Before lamination", {required: true, max: 80})}
    ${field("Later step", "after_step", "After lamination", {required: true, max: 80})}
    ${field("Earlier capture", "before_capture", ordered[0]?.id, {values: options, wide: true})}
    ${field("Later capture", "after_capture", ordered[ordered.length - 1]?.id, {values: options, wide: true})}
    </div><p class="small-copy">Both captures are rectified with their own corner placement and camera orientation,
      so a rotated camera is fine as long as the corners are placed for each original.</p>
    <p class="form-error" hidden></p><div class="form-actions">${submit("Create review")}</div></form>`);
  wireForm(root, async (values) => {
    const known = new Set((app.case.reviews || []).map((item) => item.id));
    const item = await request(app.url("/reviews"), "POST", {...values, revision: app.case.revision});
    app.modal.close();
    const created = (item.reviews || []).find((entry) => !known.has(entry.id));
    if (created) {
      view.reviewId = created.id;
      view.sel = {before: new Set(), after: new Set()};
    }
    await app.update(item, "Step review created.");
  });
}

function removeDialog(app, review) {
  const root = app.openDialog("Delete step review", `<form><p class="small-copy">Delete ${e(review.title)}?
    The confirmed region groups are removed from this inspection.</p>
    <p class="form-error" hidden></p><div class="form-actions"><button type="submit" class="danger">Delete review</button></div></form>`);
  wireForm(root, async () => {
    const item = await request(app.url(`/reviews/${review.id}`), "DELETE", {revision: app.case.revision});
    app.modal.close();
    if (view.reviewId === review.id) view.reviewId = null;
    await app.update(item, "Step review deleted.");
  });
}

function renameDialog(app, data) {
  const root = app.openDialog("Rename step review", `<form><div class="fields">
    ${field("Review title", "title", data.review.title, {required: true, wide: true, max: 120})}
    </div><p class="form-error" hidden></p><div class="form-actions">${submit("Save title")}</div></form>`);
  wireForm(root, async (values) => {
    const item = await request(app.url(`/reviews/${data.review.id}`), "PUT", {...values, revision: app.case.revision});
    app.modal.close();
    await app.update(item, "Review title saved.");
  });
}

async function renderDetail(app, root, reviewId) {
  const version = app.renderVersion;
  root.innerHTML = itemCard("Step review", '<p class="small-copy">Computing dark-region comparison…</p>');
  let data;
  try {
    data = await request(app.url(`/reviews/${reviewId}/data`));
  } catch (error) {
    if (root.isConnected) root.innerHTML = itemCard("Step review", `<p class="error">${e(error.message)}</p>`);
    return;
  }
  if (version !== app.renderVersion || !root.isConnected) return;
  const stale = data.status !== "current";
  const locked = app.case.state === "issued";
  root.innerHTML = `<div class="page-stack">
    <section class="card"><div class="card-header"><h2>${e(data.review.title)}</h2>
      <div class="toolbar">${badge(changeLabel(data.status), stale ? "training" : "")}</div></div>
      <div class="card-pad section-stack">
        ${stale ? `<p class="warning">The captures, corner placements or thresholds changed since this review was saved.
          The confirmed groups below are the saved record. Start a new review to measure again.</p>` : ""}
        <div class="review-sources">
          ${sourceCard(data.captures.before)}
          ${sourceCard(data.captures.after)}
        </div>
        <div class="toolbar">
          ${button("back-to-reviews", "All reviews", "quiet")}
          ${locked ? "" : button("rename-review", "Rename")}
          ${locked ? "" : button("delete-review", "Delete review", "danger")}
        </div>
      </div></section>
    ${stale ? "" : liveSections(app, data, locked)}
    <section class="card"><div class="card-header"><h2>Dark region groups</h2>
      ${stale ? "" : `<div class="toolbar">${locked ? "" : button("confirm-all", "Confirm all suggested", "compact", !data.groups.some((group) => !group.manual))}</div>`}</div>
      ${groupsTable(app, data, stale || locked)}
    </section>
    ${stale ? "" : inconclusiveSection(data)}
  </div>`;
  $("#back-to-reviews", root).onclick = () => { view.reviewId = null; app.render(); };
  $("#rename-review", root)?.addEventListener("click", () => renameDialog(app, data));
  $("#delete-review", root)?.addEventListener("click", () => removeDialog(app, data.review));
  if (stale) return;
  wireLiveView(app, root, data, locked);
}

function sourceCard(source) {
  if (source.missing) {
    return `<div class="card"><div class="card-pad">${definition([["Step", source.step], ["Original", "no longer in this inspection"]])}</div></div>`;
  }
  return `<div class="card"><div class="card-pad section-stack">
    <div class="toolbar">${badge(source.step, "neutral")}
      ${source.placement_explicit ? badge("Corners placed") : badge("Default full-frame placement", "training")}</div>
    ${definition([
      ["Original", source.name],
      ["Role", ROLE[source.role] ?? source.role],
      ["Drive", source.current_a == null ? "—" : `${source.current_a} A · ${source.exposure_ms} ms`],
      ["Orientation", changeLabel(source.orientation)],
      ["SHA-256", `${source.digest.slice(0, 16)}…`],
    ])}</div></div>`;
}

function liveSections(app, data, locked) {
  const s = data.summary;
  return `<section class="card"><div class="card-header"><h2>Change summary</h2>
      <span class="chip">${data.groups.length} group${data.groups.length === 1 ? "" : "s"}</span></div>
    <div class="card-pad">${metrics([
      ["New after step", s.new], ["Grown", s.grown], ["Merged", s.merged], ["Split", s.split],
      ["Shrunk", s.shrunk], ["Cleared", s.cleared], ["Steady 1:1", s.steady],
      ["Confirmed groups", s.manual_links],
      ["Δ dark area", deltaArea(s)],
      ["Cells inconclusive", s.inconclusive_cells],
    ])}</div></section>
  <div class="review-pair" id="review-viewers">
    ${["before", "after"].map((side) => `<section class="card">
      <div class="card-header"><h2>${e(data.captures[side].step)}</h2>${badge(ROLE[data.captures[side].role] ?? "", "neutral")}</div>
      <div class="viewer-well"><div class="viewer-viewport">
        <img id="review-img-${side}" alt="${e(data.captures[side].step)} dark regions"
          src="${e(app.url(`/reviews/${data.review.id}/image/${side}.png`, {revision: app.case.revision}))}">
        <canvas id="review-canvas-${side}" tabindex="0" aria-label="Select dark regions in the ${e(data.captures[side].step)} image"></canvas>
      </div></div>
      <div class="viewer-footer"><span>${e(data.captures[side].name)}</span><span id="sel-count-${side}"></span></div>
    </section>`).join("")}
  </div>
  ${locked ? "" : `<section class="card"><div class="card-header"><h2>Region linking</h2></div>
    <div class="card-pad section-stack" id="link-bar"></div></section>`}`;
}

function inconclusiveSection(data) {
  const rows = data.inconclusive_cells;
  return `<section class="card"><div class="card-header"><h2>Cells without a conclusion</h2>
    <span class="chip ${rows.length ? "warm" : ""}">${rows.length}</span></div>
    ${rows.length ? `<p class="small-copy card-pad">These cells were blocked, cropped or missing in at least one capture,
      so dark-region change cannot be concluded there. Retake or unblock the capture and start a new review to cover them.</p>
    ${table(
      [["Cell", "cell"], [`Support · ${data.review.before_step}`, "before_support", percent],
       [`Support · ${data.review.after_step}`, "after_support", percent], ["Reason", "reason"]],
      rows,
    )}` : '<p class="empty-line">Every cell has usable support in both captures.</p>'}
  </section>`;
}

function groupsTable(app, data, readonly) {
  if (!data.groups.length) return '<p class="empty-line">No dark regions in either capture.</p>';
  const rows = data.groups.map((group) => `<tr>
    <td><span class="chip" style="background:${group.color}2e;color:${group.color}">${e(group.key)}</span></td>
    <td>${badge(VERDICT_LABEL[group.verdict] ?? group.verdict, VERDICT_BADGE[group.verdict] ?? "neutral")}</td>
    <td class="num">${e(group.before.join(", ") || "—")}</td>
    <td class="num">${e(group.after.join(", ") || "—")}</td>
    <td class="num">${areaText(group.before_mm2, group.before_pixels)}</td>
    <td class="num">${areaText(group.after_mm2, group.after_pixels)}</td>
    <td class="num">${deltaText(group)}</td>
    <td class="num">${e(group.cells.join(", "))}</td>
    <td>${group.caution.length ? `<span class="warm">${e(group.caution.join("; "))}</span>` : "—"}</td>
    <td>${group.manual ? "Confirmed" : "Suggested"}</td>
    ${readonly ? "" : `<td><div class="toolbar">
      <button type="button" class="compact" data-focus="${e(group.key)}">Focus</button>
      ${group.manual
        ? `<button type="button" class="compact" data-unlink="${e(group.link_id)}">Unlink</button>`
        : `<button type="button" class="compact" data-confirm="${e(group.key)}">Confirm</button>`}
    </div></td>`}
  </tr>`).join("");
  return `<div class="table-wrap"><table><thead><tr>
    <th scope="col">Group</th><th scope="col">Assessment</th>
    <th scope="col">${e(data.review.before_step)}</th><th scope="col">${e(data.review.after_step)}</th>
    <th scope="col">Before mm²</th><th scope="col">After mm²</th><th scope="col">Δ mm²</th>
    <th scope="col">Cells</th><th scope="col">Caution</th><th scope="col">Origin</th>
    ${readonly ? "" : '<th scope="col"><span class="muted">Actions</span></th>'}
    </tr></thead><tbody>${rows}</tbody></table></div>`;
}

function areaText(mm2, pixels) {
  return mm2 == null ? `${format(pixels)} px` : format(mm2, 1);
}

function signed(value) {
  return `${value > 0 ? "+" : ""}${format(value, 1)}`;
}

function deltaText(group) {
  return group.delta_mm2 == null ? `${signed(group.delta_pixels)} px` : signed(group.delta_mm2);
}

function deltaArea(summary) {
  if (summary.delta_mm2 != null) return `${signed(summary.delta_mm2)} mm²`;
  return `${signed(summary.after_dark_pixels - summary.before_dark_pixels)} px`;
}

function wireLiveView(app, root, data, locked) {
  const membership = new Map();
  for (const group of data.groups) {
    for (const side of ["before", "after"]) {
      for (const id of group[side]) membership.set(id, group);
    }
  }
  for (const side of ["before", "after"]) {
    const image = $(`#review-img-${side}`, root);
    const canvas = $(`#review-canvas-${side}`, root);
    const redraw = bindCanvas(canvas, image, (painter, width, height) =>
      paintRegions(painter, width, height, side, data, membership));
    view.redraw[side] = redraw;
    canvas.addEventListener("click", (event) => {
      if (locked) return;
      const region = pickRegion(data, side, normalizedPoint(event, canvas));
      if (!region) return;
      if (view.sel[side].has(region.id)) view.sel[side].delete(region.id);
      else view.sel[side].add(region.id);
      refreshSelection(app, data);
    });
  }
  refreshSelection(app, data);
  $$("[data-focus]", root).forEach((node) => node.onclick = () => {
    const group = data.groups.find((entry) => entry.key === node.dataset.focus);
    if (!group) return;
    view.sel = {before: new Set(group.before), after: new Set(group.after)};
    refreshSelection(app, data);
    $("#review-viewers", root)?.scrollIntoView({behavior: "smooth", block: "nearest"});
  });
  $$("[data-confirm]", root).forEach((node) => node.onclick = () => {
    const group = data.groups.find((entry) => entry.key === node.dataset.confirm);
    if (group) saveLinks(app, data, [...manualLinks(data), linkOf(group)], `Group ${group.key} confirmed.`);
  });
  $$("[data-unlink]", root).forEach((node) => node.onclick = () =>
    saveLinks(app, data, manualLinks(data).filter((link) => link.id !== node.dataset.unlink), "Group unlinked."));
  $("#confirm-all", root)?.addEventListener("click", () => {
    const additions = data.groups.filter((group) => !group.manual).map(linkOf);
    saveLinks(app, data, [...manualLinks(data), ...additions], "All suggested groups confirmed.");
  });
}

function paintRegions(painter, width, height, side, data, membership) {
  if (!data.plane) return;
  for (const region of data.regions[side]) {
    const [x0, y0, x1, y1] = region.bounds;
    const rx = x0 / data.plane.width * width;
    const ry = y0 / data.plane.height * height;
    const rw = (x1 - x0) / data.plane.width * width;
    const rh = (y1 - y0) / data.plane.height * height;
    const selected = view.sel[side].has(region.id);
    painter.lineWidth = selected ? 3 : 1.3;
    painter.strokeStyle = selected ? "#ffd98a" : membership.get(region.id)?.color ?? "#9fb4c0";
    if (selected) {
      painter.fillStyle = "#ffd98a30";
      painter.fillRect(rx, ry, rw, rh);
    }
    painter.strokeRect(rx, ry, rw, rh);
  }
}

function pickRegion(data, side, point) {
  const x = point[0] * data.plane.width;
  const y = point[1] * data.plane.height;
  let best = null;
  for (const region of data.regions[side]) {
    const [x0, y0, x1, y1] = region.bounds;
    if (x >= x0 && x < x1 && y >= y0 && y < y1) {
      const area = (x1 - x0) * (y1 - y0);
      if (!best || area < best.area) best = {region, area};
    }
  }
  if (best) return best.region;
  let nearest = null;
  let distance = Infinity;
  for (const region of data.regions[side]) {
    const gap = Math.hypot(region.center[0] - x, region.center[1] - y);
    if (gap < distance) {
      distance = gap;
      nearest = region;
    }
  }
  return distance <= Math.max(10, data.plane.width * 0.012) ? nearest : null;
}

function refreshSelection(app, data) {
  for (const side of ["before", "after"]) {
    view.redraw[side]?.();
    const count = $(`#sel-count-${side}`);
    if (count) count.textContent = view.sel[side].size ? `${view.sel[side].size} selected` : "";
  }
  const bar = $("#link-bar");
  if (!bar) return;
  const before = view.sel.before.size;
  const after = view.sel.after.size;
  bar.innerHTML = `<p class="small-copy">Click dark regions in either image, then link the two selections into one
    group — for example when neighbouring spots merged or one region split. The group reports the true area change.</p>
    <div class="toolbar">
      ${button("make-link", `Link ${before} before ↔ ${after} after`, "primary", before + after === 0)}
      ${button("clear-sel", "Clear selection", "compact", before + after === 0)}
    </div>
    <div class="chips">${["before", "after"].map((side) => [...view.sel[side]].map((id) =>
      `<button type="button" class="chip" data-unselect="${e(side)}:${e(id)}" title="Remove from selection">${e(id)} ✕</button>`
    ).join("")).join("")}</div>`;
  $("#make-link", bar).onclick = () => {
    const links = manualLinks(data);
    links.push({before: [...view.sel.before], after: [...view.sel.after], note: ""});
    saveLinks(app, data, links, "Regions linked into a group.");
  };
  $("#clear-sel", bar).onclick = () => {
    view.sel = {before: new Set(), after: new Set()};
    refreshSelection(app, data);
  };
  $$("[data-unselect]", bar).forEach((node) => node.onclick = () => {
    const [side, id] = node.dataset.unselect.split(":");
    view.sel[side].delete(id);
    refreshSelection(app, data);
  });
}

function linkOf(group) {
  return {before: group.before, after: group.after, note: group.note || ""};
}

function manualLinks(data) {
  return data.groups.filter((group) => group.manual).map((group) => ({id: group.link_id, ...linkOf(group)}));
}

async function saveLinks(app, data, links, message) {
  const payload = links.map(({before, after, note}) => ({before, after, note}));
  try {
    const item = await request(app.url(`/reviews/${data.review.id}`), "PUT", {links: payload, revision: app.case.revision});
    view.sel = {before: new Set(), after: new Set()};
    await app.update(item, message);
  } catch (error) {
    app.fail(error);
  }
}
