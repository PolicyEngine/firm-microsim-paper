import { SlideshowConfig } from "@/lib/types";
import {
  AppendixBehavSlide,
  AppendixGenSlide,
  AppendixNotchSlide,
  AppendixSecondarySlide,
  BehaviouralLayerSlide,
  BunchingMechanicalSlide,
  ConclusionSlide,
  DataSlide,
  DominatedRegionSlide,
  FirmProblemSlide,
  FirmsClusterSlide,
  LiteratureSlide,
  QuestionSlide,
  ReformMenuSlide,
  SectionBehavioural,
  SectionConclusion,
  SectionData,
  SectionDistortion,
  SectionMotivation,
  SectionStatic,
  StaticCostingSlide,
  StaticSweepSlide,
  ThankYouSlide,
  ThisPaperSlide,
  TitleSlide,
  TurnoverDistSlide,
  ValidationSlide,
  WhichReformsSlide,
  WhyThresholdSlide,
} from "@/slides/vat-ima-2026";

export const vatIma2026Config: SlideshowConfig = {
  id: "vat-ima-2026",
  title: "An Open Firm-Level Microsimulation of the UK VAT Registration Threshold",
  description:
    "PolicyEngine conference deck for the IMA World Congress 2026 — costing UK VAT registration-threshold reforms on synthetic firm data.",
  date: "2026-07-01",
  location: "IMA World Congress 2026, Brussels",
  footerText: "IMA World Congress 2026 · Firm microsimulation",
  speakers: [{ name: "Vahid Ahmadi", title: "PolicyEngine" }],
  // Slides 0–24 are the main deck; 25+ are appendix (excluded from the counter).
  mainSlideCount: 25,
  slides: [
    // 0 · Open
    TitleSlide,
    // 1 · Motivation
    SectionMotivation,
    QuestionSlide,
    WhyThresholdSlide,
    FirmsClusterSlide,
    ThisPaperSlide,
    LiteratureSlide,
    // 2 · Data and calibration
    SectionData,
    DataSlide,
    TurnoverDistSlide,
    // 3 · Static costing
    SectionStatic,
    StaticCostingSlide,
    ValidationSlide,
    StaticSweepSlide,
    ReformMenuSlide,
    // 4 · The notch's distortion
    SectionDistortion,
    BunchingMechanicalSlide,
    DominatedRegionSlide,
    WhichReformsSlide,
    // 5 · Behavioural layer
    SectionBehavioural,
    FirmProblemSlide,
    BehaviouralLayerSlide,
    // 6 · Conclusion
    SectionConclusion,
    ConclusionSlide,
    ThankYouSlide,
    // Appendix (linear, after the main deck — excluded from the page counter)
    AppendixNotchSlide,
    AppendixGenSlide,
    AppendixBehavSlide,
    AppendixSecondarySlide,
  ],
};
