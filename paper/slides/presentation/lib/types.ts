import { ComponentType } from "react";

export interface SpeakerInfo {
  name: string;
  title: string;
  /** Optional headshot image path (under /public); falls back to initials. */
  headshot?: string;
}

export interface SlideEntry {
  component: ComponentType;
}

export type SlideConfig = ComponentType | SlideEntry;

export interface SlideshowConfig {
  id: string;
  title: string;
  description: string;
  date: string;
  location?: string;
  footerText?: string;
  speakers?: SpeakerInfo[];
  slides: SlideConfig[];
  /**
   * Number of leading "main" slides. Slides at or beyond this index are
   * appendix slides and are excluded from the page counter.
   */
  mainSlideCount?: number;
}

export function formatDate(iso: string): string {
  const [year, month, day] = iso.split("-").map(Number);
  const monthNames = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];
  const monthName = monthNames[month - 1] ?? "";
  return day === 1 ? `${monthName} ${year}` : `${monthName} ${day}, ${year}`;
}

export function getSlideComponent(slide: SlideConfig): ComponentType {
  if (typeof slide === "object" && "component" in slide) {
    return slide.component;
  }
  return slide;
}
