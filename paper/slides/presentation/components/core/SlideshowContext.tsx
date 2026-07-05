"use client";

import { createContext, ReactNode, useContext } from "react";

export interface SlideshowContextValue {
  id: string;
  title: string;
  date: string;
  location?: string;
  footerText: string;
  currentSlide: number;
  totalSlides: number;
  /** Main-deck slide count; slides at or past this index are appendix. */
  mainSlideCount?: number;
  /** True when rendering for PDF export (?export=1) — navigation chrome is hidden. */
  isExport: boolean;
  /** Jump to a slide by zero-based index (used by in-deck appendix links). */
  goToSlide: (index: number) => void;
}

const SlideshowContext = createContext<SlideshowContextValue | null>(null);

export function SlideshowProvider({
  children,
  value,
}: {
  children: ReactNode;
  value: SlideshowContextValue;
}) {
  return (
    <SlideshowContext.Provider value={value}>
      {children}
    </SlideshowContext.Provider>
  );
}

export function useSlideshowContextSafe() {
  return useContext(SlideshowContext);
}
