"use strict";
document.addEventListener("DOMContentLoaded", () => {
  const area = document.getElementById("skills");
  const input = document.getElementById("skill-search");
  const picker = document.getElementById("skill-suggestions");
  const add = document.getElementById("add-skill");
  const list = document.getElementById("skill-chip-list");
  if (!area || !input || !picker || !add || !list) return;

  const entries = () => area.value.split(",").map((item) => item.trim()).filter(Boolean);
  const render = () => {
    list.replaceChildren();
    entries().forEach((skill, index) => {
      const pill = document.createElement("span");
      pill.className = "skill-chip";
      pill.append(document.createTextNode(skill + " "));
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", "Remove " + skill);
      remove.addEventListener("click", () => {
        area.value = entries().filter((_, position) => position !== index).join(", ");
        render();
        input.focus();
      });
      pill.append(remove);
      list.append(pill);
    });
  };

  const addSkill = () => {
    const value = input.value.trim();
    const items = entries();
    if (!value || value.includes(",") || value.length > 100) return;
    if (items.length >= 30 || items.some((item) => item.toLocaleLowerCase() === value.toLocaleLowerCase())) return;
    const proposed = [...items, value].join(", ");
    if (proposed.length > 500) return;
    area.value = proposed;
    input.value = "";
    picker.replaceChildren();
    render();
    area.dispatchEvent(new Event("input", { bubbles: true }));
  };

  add.addEventListener("click", addSkill);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addSkill();
    }
  });
  area.addEventListener("input", render);

  let sequence = 0;
  input.addEventListener("input", async () => {
    const query = input.value.trim();
    const current = ++sequence;
    picker.replaceChildren();
    if (query.length < 2 || query.length > 80) return;
    try {
      const response = await fetch("/lookup/skills?q=" + encodeURIComponent(query),
        { credentials: "same-origin" });
      if (!response.ok) return;
      const choices = await response.json();
      if (current !== sequence) return;
      choices.forEach((entry) => {
        const option = document.createElement("option");
        option.value = entry.label;
        picker.append(option);
      });
    } catch {
      // Free-text skills still work without suggestions.
    }
  });
  render();
});
