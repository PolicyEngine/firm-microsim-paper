"use client";

import { Suspense, createElement, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { SlideshowProvider } from "@/components/core/SlideshowContext";
import { buildSlideUrl } from "@/lib/slide-url";
import { getSlideComponent, SlideshowConfig } from "@/lib/types";

interface SlideshowViewerProps {
  config: SlideshowConfig;
}

function SlideshowViewerClient({ config }: SlideshowViewerProps) {
  const searchParams = useSearchParams();
  const slides = useMemo(() => config.slides, [config.slides]);
  const isExport = searchParams.get("export") === "1";
  const [currentSlide, setCurrentSlide] = useState(() => {
    const initialSlide = Number.parseInt(searchParams.get("slide") || "0", 10);
    return Math.max(0, Math.min(initialSlide, slides.length - 1));
  });
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  useEffect(() => {
    const url = buildSlideUrl(window.location.href, currentSlide);
    window.history.replaceState(null, "", url);
  }, [currentSlide]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "ArrowRight" || event.key === " ") {
        event.preventDefault();
        setCurrentSlide((previous) => Math.min(previous + 1, slides.length - 1));
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        setCurrentSlide((previous) => Math.max(previous - 1, 0));
      } else if (event.key === "PageDown") {
        // Conference presenter clickers emit PageDown/PageUp.
        event.preventDefault();
        setCurrentSlide((previous) => Math.min(previous + 1, slides.length - 1));
      } else if (event.key === "PageUp") {
        event.preventDefault();
        setCurrentSlide((previous) => Math.max(previous - 1, 0));
      } else if (event.key === "Home") {
        event.preventDefault();
        setCurrentSlide(0);
      } else if (event.key === "End") {
        event.preventDefault();
        setCurrentSlide(slides.length - 1);
      } else if (event.key === "f" || event.key === "F11") {
        event.preventDefault();
        if (document.fullscreenElement) {
          void document.exitFullscreen();
        } else {
          void document.documentElement.requestFullscreen();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [slides.length]);

  const currentSlideElement = createElement(getSlideComponent(slides[currentSlide]));
  const footerText = config.footerText ?? `${config.id.replace(/-/g, " ")} - ${config.date}`;

  return (
    <SlideshowProvider
      value={{
        id: config.id,
        title: config.title,
        date: config.date,
        location: config.location,
        footerText,
        currentSlide,
        totalSlides: slides.length,
        goToSlide: (index: number) =>
          setCurrentSlide(Math.max(0, Math.min(index, slides.length - 1))),
      }}
    >
      <main
        className="relative cursor-pointer"
        onClick={() => setCurrentSlide((previous) => Math.min(previous + 1, slides.length - 1))}
      >
        <div className="slide-active">{currentSlideElement}</div>

        {!isFullscreen && !isExport && (
          <div className="pointer-events-none fixed bottom-0 left-0 right-0 z-50 flex h-18 items-center justify-end gap-4 px-8 text-white">
            <button
              aria-label="Previous slide"
              className="pointer-events-auto flex h-10 w-10 items-center justify-center text-2xl transition hover:opacity-75"
              onClick={(event) => {
                event.stopPropagation();
                setCurrentSlide((previous) => Math.max(previous - 1, 0));
              }}
              type="button"
            >
              {"<"}
            </button>
            <span className="min-w-16 text-center text-sm font-semibold">
              {currentSlide < (config.mainSlideCount ?? slides.length)
                ? `${currentSlide + 1} / ${config.mainSlideCount ?? slides.length}`
                : "Appendix"}
            </span>
            <button
              aria-label="Next slide"
              className="pointer-events-auto flex h-10 w-10 items-center justify-center text-2xl transition hover:opacity-75"
              onClick={(event) => {
                event.stopPropagation();
                setCurrentSlide((previous) => Math.min(previous + 1, slides.length - 1));
              }}
              type="button"
            >
              {">"}
            </button>
          </div>
        )}
      </main>
    </SlideshowProvider>
  );
}

export default function SlideshowViewer(props: SlideshowViewerProps) {
  return (
    <Suspense fallback={<div className="min-h-screen bg-white" />}>
      <SlideshowViewerClient {...props} />
    </Suspense>
  );
}
