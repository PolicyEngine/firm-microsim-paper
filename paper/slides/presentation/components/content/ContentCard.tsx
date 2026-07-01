import { ReactNode } from "react";

interface ContentCardProps {
  title?: string;
  children: ReactNode;
  accent?: "teal" | "amber" | "slate";
  className?: string;
}

export default function ContentCard({
  title,
  children,
  accent = "teal",
  className = "",
}: ContentCardProps) {
  const accentClass =
    accent === "amber"
      ? "border-t-pe-amber"
      : accent === "slate"
        ? "border-t-slate-400"
        : "border-t-pe-teal";

  return (
    <div
      className={`rounded-lg border border-slate-200 border-t-4 ${accentClass} bg-white p-7 shadow-sm ${className}`}
    >
      {title && <h2 className="mb-4 text-2xl font-bold text-pe-dark">{title}</h2>}
      {children}
    </div>
  );
}
