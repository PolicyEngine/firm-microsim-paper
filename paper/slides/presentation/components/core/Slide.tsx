"use client";

import { ReactNode } from "react";
import Image from "@/components/core/BasePathImage";
import { useSlideshowContextSafe } from "@/components/core/SlideshowContext";

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
          <div className="h-full w-full">{children}</div>
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
