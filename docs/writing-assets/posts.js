(() => {
  const status = document.getElementById("writing-status");
  const posts = document.getElementById("writing-posts");
  const source = "https://github.com/858006908cc/cc.github.io/blob/main/docs/writing.md";

  function showFallback() {
    const link = document.createElement("a");
    link.href = source;
    link.textContent = "Read posts on GitHub";
    status.replaceChildren(link, ".");
  }

  async function loadPosts() {
    try {
      if (typeof window.markdownit !== "function") {
        throw new Error("Markdown renderer unavailable");
      }
      const response = await fetch(new URL("writing.md", document.baseURI), { cache: "no-cache" });
      if (!response.ok) {
        throw new Error(`Could not load posts (${response.status})`);
      }
      const markdown = (await response.text()).replace(/<!--[\s\S]*?-->/g, "").trim();
      if (!markdown) {
        return;
      }
      const renderer = window.markdownit({ html: false });
      const headingOpen = renderer.renderer.rules.heading_open;
      const headingClose = renderer.renderer.rules.heading_close;
      for (const name of ["heading_open", "heading_close"]) {
        const original = name === "heading_open" ? headingOpen : headingClose;
        renderer.renderer.rules[name] = (tokens, index, options, env, self) => {
          const token = tokens[index];
          const level = Number(token.tag.slice(1));
          token.tag = "h" + Math.min(6, level + 1);
          return original
            ? original(tokens, index, options, env, self)
            : self.renderToken(tokens, index, options);
        };
      }
      posts.innerHTML = renderer.render(markdown);
      status.hidden = true;
    } catch {
      showFallback();
    }
  }

  void loadPosts();
})();
