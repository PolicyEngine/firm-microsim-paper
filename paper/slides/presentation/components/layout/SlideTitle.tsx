import { ReactNode } from "react";

interface SlideTitleProps {
  children: ReactNode;
  kicker?: string;
  className?: string;
}

export default function SlideTitle({ children, kicker, className = "" }: SlideTitleProps) {
  return (
    <div className={className}>
      {kicker && (
        <div className="mb-3 text-sm font-bold uppercase tracking-[0.22em] text-pe-teal">
          {kicker}
        </div>
      )}
      <h1 className="max-w-5xl text-5xl font-extrabold leading-tight tracking-tight text-pe-dark">
        {children}
      </h1>
    </div>
  );
}
