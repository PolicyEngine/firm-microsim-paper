"use client";

import SlideshowViewer from "@/components/core/SlideshowViewer";
import { vatIma2026Config } from "@/slides/config";

export default function Home() {
  return <SlideshowViewer config={vatIma2026Config} />;
}
