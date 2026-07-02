"use client";

import { ReactNode, useLayoutEffect, useRef } from "react";
import Image from "@/components/core/BasePathImage";
import { useSlideshowContextSafe } from "@/components/core/SlideshowContext";

// Shrink-to-fit safety net: if the slide content is taller than the available
// content box (e.g. a short browser window), scale the content down just enough
// to fit — white background still fills the window and the footer stays put.
// At normal 16:9 window sizes the content fits and no transform is applied, so
// the DOM and visuals are identical to an unscaled slide.
function useFitToContentBox() {
  const contentRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const element = contentRef.current;
    if (!element) return;

    const update = () => {
      // Measure at natural scale (transforms do not affect layout, but they do
      // affect client rects, so clear ours before measuring). This runs before
      // paint, so the reset is never visible.
      element.style.transform = "";
      const box = element.getBoundingClientRect();
      if (box.height === 0) return;

      // Content extent relative to the content box, including anything that
      // spills above it (centered layouts overflow both ways).
      let minTop = 0;
      let maxBottom = box.height;
      for (const descendant of element.querySelectorAll("*")) {
        const rect = descendant.getBoundingClientRect();
        if (rect.height === 0 && rect.width === 0) continue;
        minTop = Math.min(minTop, rect.top - box.top);
        maxBottom = Math.max(maxBottom, rect.bottom - box.top);
      }

      const needed = maxBottom - minTop;
      if (needed > box.height + 1) {
        const scale = box.height / needed;
        // Anchor the (scaled) content extent to the top of the content box and
        // re-centre it horizontally after the width shrinks.
        const translateX = ((1 - scale) * box.width) / 2;
        element.style.transformOrigin = "top left";
        element.style.transform = `translate(${translateX}px, ${-scale * minTop}px) scale(${scale})`;
      }
    };

    update();
    // Re-fit when the window changes and once webfonts have swapped in.
    window.addEventListener("resize", update);
    document.fonts?.ready.then(update).catch(() => undefined);
    return () => window.removeEventListener("resize", update);
  }, []);

  return contentRef;
}

interface SlideProps {
  children: ReactNode;
  className?: string;
  isCover?: boolean;
  isEnd?: boolean;
  showFooter?: boolean;
  fullBleed?: boolean;
}

export default function Slide({
  children,
  className = "",
  isCover = false,
  isEnd = false,
  showFooter = true,
  fullBleed = false,
}: SlideProps) {
  const context = useSlideshowContextSafe();
  const footerText = context?.footerText ?? "";
  const contentRef = useFitToContentBox();

  return (
    <section
      className={[
        "relative flex h-screen w-screen flex-col overflow-hidden",
        isCover || isEnd ? "gradient-bg items-center justify-center text-white" : "bg-white",
        className,
      ].join(" ")}
    >
      {fullBleed ? (
        <div className="absolute inset-0">{children}</div>
      ) : (
        <div
          className={[
            "absolute inset-0 z-10",
            isCover || isEnd ? "flex items-center justify-center px-20" : "px-16 pb-28 pt-20",
          ].join(" ")}
        >
          <div className="h-full w-full" ref={contentRef}>
            {children}
          </div>
        </div>
      )}

      {showFooter && !isCover && !isEnd && (
        <footer className="gradient-footer absolute bottom-0 left-0 right-0 z-20 flex h-18 items-center px-16 text-white">
          <Image
            alt="PolicyEngine"
            className="opacity-90"
            height={50}
            src="/logos/white.svg"
            style={{ height: "auto" }}
            width={180}
          />
          <div className="pointer-events-none absolute left-1/2 -translate-x-1/2 text-sm font-medium opacity-90">
            {footerText}
          </div>
        </footer>
      )}
    </section>
  );
}
