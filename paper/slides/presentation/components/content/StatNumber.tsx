interface StatNumberProps {
  value: string;
  label: string;
  sublabel?: string;
  compact?: boolean;
}

export default function StatNumber({ value, label, sublabel, compact = false }: StatNumberProps) {
  return (
    <div className={["rounded-lg border border-slate-200 bg-white text-center shadow-sm", compact ? "p-4" : "p-7"].join(" ")}>
      <div className={[compact ? "text-4xl" : "text-5xl", "font-extrabold tracking-tight text-pe-teal"].join(" ")}>
        {value}
      </div>
      <div className={[compact ? "mt-2 text-lg" : "mt-3 text-xl", "font-bold text-pe-dark"].join(" ")}>
        {label}
      </div>
      {sublabel && <div className={[compact ? "mt-1 text-sm" : "mt-2 text-base", "text-slate-500"].join(" ")}>{sublabel}</div>}
    </div>
  );
}
