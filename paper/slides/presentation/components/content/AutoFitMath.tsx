"use client";

import { ReactNode, useLayoutEffect, useRef, useState } from "react";

/**
 * Scales its child down (never up) so wide content fits the available width.
 * Used for KaTeX equations on slides, where horizontal scrolling is not an
 * option during a live talk. Measurement uses offsetWidth, which is the
 * untransformed layout width, so applying the transform never feeds back.
 */
export default function AutoFitMath({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  const outerRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);

  useLayoutEffect(() => {
    const outer = outerRef.current;
    const inner = innerRef.current;
    if (!outer || !inner) return;

    const fit = () => {
      const available = outer.clientWidth;
      const natural = inner.offsetWidth;
      if (natural > 0 && available > 0) {
        setScale(natural > available ? available / natural : 1);
      }
    };

    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(outer);
    // KaTeX fonts can change the measured width once they load.
    if (typeof document !== "undefined" && document.fonts?.ready) {
      document.fonts.ready.then(fit).catch(() => {});
    }
    return () => ro.disconnect();
  }, [children]);

  return (
    <div ref={outerRef} className={`w-full overflow-hidden text-center ${className}`}>
      <div
        ref={innerRef}
        className="inline-block"
        style={{ transform: `scale(${scale})`, transformOrigin: "center" }}
      >
        {children}
      </div>
    </div>
  );
}
