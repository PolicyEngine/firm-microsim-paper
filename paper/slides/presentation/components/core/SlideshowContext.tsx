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
