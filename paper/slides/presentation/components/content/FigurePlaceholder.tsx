interface FigurePlaceholderProps {
  title: string;
  subtitle: string;
  kind?: "frontier" | "bars" | "tradeoff";
}

export default function FigurePlaceholder({
  title,
  subtitle,
  kind = "frontier",
}: FigurePlaceholderProps) {
  const lines =
    kind === "bars"
      ? "items-end"
      : kind === "tradeoff"
        ? "items-center"
        : "items-end";

  return (
    <div className="figure-grid flex h-full min-h-[420px] flex-col rounded-lg border border-slate-200 bg-white p-8 shadow-sm">
      <div>
        <div className="text-2xl font-bold text-pe-dark">{title}</div>
        <div className="mt-2 text-lg text-slate-500">{subtitle}</div>
      </div>
      <div className={`mt-10 grid flex-1 grid-cols-5 gap-6 ${lines}`}>
        {[38, 56, 74, 64, 86].map((height, index) => (
          <div key={height} className="flex h-full flex-col justify-end gap-3">
            <div
              className={[
                "rounded-t-md",
                index % 3 === 0 ? "bg-pe-teal" : index % 3 === 1 ? "bg-sky-500" : "bg-slate-400",
              ].join(" ")}
              style={{ height: `${height}%` }}
            />
            <div className="h-3 rounded-full bg-slate-200" />
          </div>
        ))}
      </div>
      <div className="mt-8 text-sm font-semibold uppercase tracking-[0.18em] text-slate-400">
        Figure pending final sweep outputs
      </div>
    </div>
  );
}
