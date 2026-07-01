import katex from "katex";

interface MathProps {
  tex: string;
  display?: boolean;
  className?: string;
}

// Server-rendered KaTeX. renderToString is pure JS, so this stays a server
// component and the typeset markup ships in the initial HTML (no hydration flash).
export default function Math({ tex, display = false, className = "" }: MathProps) {
  const html = katex.renderToString(tex, {
    displayMode: display,
    throwOnError: false,
    strict: false,
    trust: false,
  });

  if (display) {
    return (
      <div
        className={className}
        // KaTeX output is trusted markup generated from our own TeX strings.
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }

  return (
    <span
      className={className}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
