"use client";

import { ReactNode } from "react";

import AutoFitMath from "@/components/content/AutoFitMath";
import Figure from "@/components/content/Figure";
import Math from "@/components/content/Math";
import Image from "@/components/core/BasePathImage";
import Slide from "@/components/core/Slide";
import { useSlideshowContextSafe } from "@/components/core/SlideshowContext";
import CoverSlide from "@/components/layout/CoverSlide";
import SectionSlide from "@/components/layout/SectionSlide";
import SlideTitle from "@/components/layout/SlideTitle";

/**
 * Zero-based slide indices, kept in sync with the order in `slides/config.ts`.
 * Used to wire the in-deck appendix jump links (forward) and back links.
 */
const SLIDE = {
  whyThreshold: 3,
  data: 8,
  reformMenu: 14,
  dominated: 17,
  conclusion: 23,
  appNotch: 25,
  appGen: 26,
  appBehav: 27,
  appSecondary: 28,
} as const;

/* ------------------------------------------------------------------ */
/* Shared helpers                                                      */
/* ------------------------------------------------------------------ */

const speakers = [
  { name: "Vahid Ahmadi", title: "PolicyEngine", headshot: "/headshots/vahid-ahmadi.png" },
];

/** Teal-dot bullet list; items may be plain text or rich JSX (inline Math). */
function BList({
  items,
  className = "",
  size = "text-xl",
}: {
  items: ReactNode[];
  className?: string;
  size?: string;
}) {
  return (
    <ul className={`space-y-4 ${className}`}>
      {items.map((item, i) => (
        <li key={i} className={`flex gap-4 ${size} leading-snug text-slate-700`}>
          <span className="mt-2.5 h-2.5 w-2.5 flex-none rounded-full bg-pe-teal" />
          <div className="flex-1">{item}</div>
        </li>
      ))}
    </ul>
  );
}

/** Indented, muted sub-bullets under a BList item. */
function SubList({ items }: { items: ReactNode[] }) {
  return (
    <ul className="mt-2 space-y-1.5">
      {items.map((item, i) => (
        <li key={i} className="flex gap-3 text-lg leading-snug text-slate-500">
          <span className="mt-2 h-1.5 w-1.5 flex-none rounded-full bg-slate-300" />
          <div>{item}</div>
        </li>
      ))}
    </ul>
  );
}

function Caption({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <p className={`text-base leading-snug text-slate-500 ${className}`}>{children}</p>;
}

/** Auto-fitting centred display equation. */
function DisplayEq({ tex, className = "" }: { tex: string; className?: string }) {
  return (
    <AutoFitMath className={className}>
      <Math tex={tex} display className="text-slate-700" />
    </AutoFitMath>
  );
}

/** Clickable pointer that jumps to a matching appendix slide. */
function SeeAppendix({ to, children }: { to: number; children: ReactNode }) {
  const ctx = useSlideshowContextSafe();
  return (
    <button
      type="button"
      onClick={(event) => {
        event.stopPropagation();
        ctx?.goToSlide(to);
      }}
      className="pointer-events-auto mt-6 text-sm font-medium uppercase tracking-[0.14em] text-pe-teal/80 transition hover:text-pe-teal hover:underline"
    >
      → {children}
    </button>
  );
}

/** Clickable "back" link on an appendix slide that returns to its source slide. */
function BackTo({ to, children }: { to: number; children: ReactNode }) {
  const ctx = useSlideshowContextSafe();
  return (
    <button
      type="button"
      onClick={(event) => {
        event.stopPropagation();
        ctx?.goToSlide(to);
      }}
      className="pointer-events-auto mt-8 text-sm font-medium uppercase tracking-[0.14em] text-pe-teal/80 transition hover:text-pe-teal hover:underline"
    >
      ← {children}
    </button>
  );
}

const teal = (children: ReactNode) => (
  <span className="font-semibold text-pe-teal">{children}</span>
);

/* ================================================================== */
/* 0 · Open                                                            */
/* ================================================================== */

export function TitleSlide() {
  return (
    <CoverSlide
      title="An Open Firm-Level Microsimulation of the UK VAT Registration Threshold"
      event="IMA World Congress 2026 · Brussels · Parallel Session 4A"
      date="2026-07-01"
      speakers={speakers}
    />
  );
}

/* ================================================================== */
/* 1 · Motivation                                                      */
/* ================================================================== */

export function SectionMotivation() {
  return (
    <SectionSlide
      section="01 · Motivation"
      title="Motivation"
      subtitle="Why the UK VAT threshold is a notch — and how to choose the more efficient policy."
    />
  );
}

export function QuestionSlide() {
  return (
    <Slide>
      <div className="flex h-full flex-col justify-center">
        <SlideTitle kicker="The question" className="mb-10">
          The question
        </SlideTitle>
        <p className="max-w-5xl text-4xl font-semibold leading-snug text-slate-800">
          How should the UK {teal("cost reforms")} to its VAT registration threshold — changes to
          its {teal("level")}, {teal("shape")}, or {teal("rate")} — on a{" "}
          {teal("single, common, reproducible basis")},
        </p>
        <p className="mt-6 max-w-5xl text-4xl font-semibold leading-snug text-slate-800">
          when the firm-level data needed are {teal("not openly available")}?
        </p>
      </div>
    </Slide>
  );
}

export function WhyThresholdSlide() {
  return (
    <Slide>
      <div className="flex h-full flex-col justify-center">
        <SlideTitle kicker="Motivation">Why the VAT threshold matters</SlideTitle>
        <BList
          className="mt-8"
          items={[
            <>
              The UK VAT threshold is a {teal("notch, not a kink")}: cross it and 20% falls due on
              the firm’s <em>entire</em> turnover — not just the slice above.
            </>,
            <>
              A discrete liability jump of <Math tex="\tau T^{*}=\text{£}17{,}000" /> (
              <Math tex="=20\%\times\text{£}85{,}000" />) at the threshold.
            </>,
            <>
              Threshold was {teal("£85,000")} (2023–24), raised to {teal("£90,000")} on 1 April 2024
              — after a seven-year nominal freeze, the longest in the tax’s history.
            </>,
            <>
              Among the highest thresholds in the OECD, and it sits in a <em>dense</em> part of the
              firm-size distribution.
            </>,
            <>
              Firms respond by {teal("bunching just below")} it; the cliff-edge keeps the threshold
              under recurrent pressure for reform.
            </>,
          ]}
        />
        <SeeAppendix to={SLIDE.appNotch}>notch vs. kink — appendix</SeeAppendix>
      </div>
    </Slide>
  );
}

export function FirmsClusterSlide() {
  const figs = [
    {
      src: "/figures/firms_by_turnover_band_85k.png",
      w: 2636,
      h: 1784,
      cap: "VAT/PAYE-registered businesses by turnover band (ONS)",
    },
    {
      src: "/figures/vat_firms_by_turnover_band_85k.png",
      w: 2636,
      h: 1784,
      cap: "VAT-registered firms by band (HMRC)",
    },
    {
      src: "/figures/obr_vat_bunching_85k.png",
      w: 2969,
      h: 1762,
      cap: "Bunching below £85k (OBR, 2023)",
    },
  ];
  return (
    <Slide>
      <div className="flex h-full flex-col">
        <SlideTitle kicker="Motivation">Firms cluster just below the threshold</SlideTitle>
        <Caption className="mt-4 max-w-6xl">
          Cross {teal("£85,000")} and 20% VAT falls due on the firm’s <em>entire</em> turnover, so
          firms cluster {teal("right below it")}.
        </Caption>
        <div className="mt-4 grid min-h-0 flex-1 grid-cols-3 grid-rows-1 gap-8">
          {figs.map((f) => (
            <div key={f.src} className="flex min-h-0 flex-col">
              <div className="min-h-0 flex-1">
                <Figure src={f.src} alt={f.cap} width={f.w} height={f.h} />
              </div>
              <Caption className="mt-3 text-center">{f.cap}</Caption>
            </div>
          ))}
        </div>
      </div>
    </Slide>
  );
}

export function ThisPaperSlide() {
  return (
    <Slide>
      <div className="flex h-full flex-col justify-center">
        <SlideTitle kicker="Motivation">This project</SlideTitle>
        <BList
          className="mt-8"
          items={[
            <>
              An {teal("open, reproducible firm-level microsimulation")} that costs UK VAT threshold
              reforms on a common static basis.
            </>,
            <>
              Built on {teal("synthetic firm microdata")} calibrated to published HMRC + ONS
              aggregates — released openly and regenerable under any counterfactual schedule.
            </>,
            <>
              Prices three reform families — {teal("level")} (move the threshold), {teal("shape")} (a
              graduated taper), {teal("rate")} (a reduced-rate band) — {teal("three ways")}:
              <SubList
                items={[
                  <>statically (mechanical reclassification);</>,
                  <>
                    by the elasticity-free {teal("dominated-region")} distortion each removes;
                  </>,
                  <>
                    behaviourally, conditional on an assumed turnover elasticity <Math tex="e" />.
                  </>,
                ]}
              />
            </>,
            <>
              <span className="font-semibold text-pe-dark">Headline:</span> only the graduated taper
              removes the notch’s dominated region (−£550m, against −£497m for a 10% band that
              removes none of it); a level rise carries the largest static cost (−£783m), relocates
              the region — and has {teal("exactly zero intensive-margin behavioural offset")}.
            </>,
          ]}
        />
      </div>
    </Slide>
  );
}

export function LiteratureSlide() {
  const rows: [ReactNode, string][] = [
    [
      <>Bunching / notch methodology — source of the “dominated region”</>,
      "Chetty et al. (2011); Saez (2010); Kleven and Waseem (2013); Kleven (2016)",
    ],
    [
      <>VAT registration threshold (empirical)</>,
      "Liu et al. (2021) (UK notch, ~43% voluntary registration); Liu, Lockwood and Tam (2024); Onji (2009); Asatryan and Peichl (2017); Waseem (2024)",
    ],
    [
      <>Does bunching identify a structural elasticity?</>,
      "Blomquist et al. (2021); Bertanha et al. (2023) — not non-parametrically identified",
    ],
    [
      <>Structural models of size-based thresholds</>,
      "Garicano et al. (2016); Gourio and Roys (2014)",
    ],
  ];
  return (
    <Slide>
      <div className="flex h-full flex-col justify-center">
        <SlideTitle kicker="Motivation">Related literature</SlideTitle>
        <BList
          className="mt-8"
          items={rows.map(([head, cites]) => (
            <>
              <span className="font-semibold text-pe-dark">{head}</span>
              <div className="mt-1 text-lg text-slate-500">{cites}</div>
            </>
          ))}
        />
      </div>
    </Slide>
  );
}

/* ================================================================== */
/* 2 · Data and calibration                                            */
/* ================================================================== */

export function SectionData() {
  return (
    <SectionSlide
      section="02 · Data & calibration"
      title="Data and calibration"
      subtitle="Synthetic firms calibrated to published HMRC and ONS aggregates."
    />
  );
}

export function DataSlide() {
  return (
    <Slide>
      <div className="grid h-full grid-cols-[1.4fr_1fr] items-center gap-12">
        <div>
          <SlideTitle kicker="Data">
            Synthetic firms calibrated to public aggregates
          </SlideTitle>
          <BList
            className="mt-8"
            size="text-lg"
            items={[
              <>
                {teal("No open firm-level VAT microdata exist")} <Math tex="\Rightarrow" /> build a
                synthetic population calibrated to published aggregates (data year 2023–24).
              </>,
              <>
                Calibration targets:
                <SubList
                  items={[
                    <>ONS <em>UK Business: Activity, Size and Location</em> — firm counts by sector × turnover band</>,
                    <>HMRC <em>VAT Annual Statistics</em> — registered counts by band / sector, net VAT liability by band</>,
                    <>OBR <em>EFO</em> (Mar 2023) Chart C — £1k-band counts £65k–£90k, pinning the near-threshold bunching profile</>,
                  ]}
                />
              </>,
              <>Mirrors open tax-benefit microsimulation (Ghenis et al., 2022).</>,
            ]}
          />
          <SeeAppendix to={SLIDE.appGen}>how firms are built — appendix</SeeAppendix>
        </div>
        <div className="space-y-4">
          <Stat value="≈2.94m" label="firm records" sub="weighted to ≈2.5m registered businesses" />
          <Stat value="89.4%" label="calibration accuracy" sub="across five calibrated dimensions" />
          <Stat value="2023–24" label="data year" sub="ONS + HMRC published aggregates" />
        </div>
      </div>
    </Slide>
  );
}

function Stat({ value, label, sub }: { value: string; label: string; sub: string }) {
  return (
    <div className="rounded-lg border border-slate-200 border-t-4 border-t-pe-teal bg-white p-5 text-center shadow-sm">
      <div className="text-4xl font-extrabold tracking-tight text-pe-teal">{value}</div>
      <div className="mt-2 text-lg font-bold text-pe-dark">{label}</div>
      <div className="mt-1 text-sm text-slate-500">{sub}</div>
    </div>
  );
}

export function TurnoverDistSlide() {
  return (
    <Slide>
      <div className="flex h-full flex-col">
        <SlideTitle kicker="Data">The calibrated turnover distribution</SlideTitle>
        <div className="mt-4 min-h-0 flex-1">
          <Figure
            src="/figures/turnover_distribution_85k.png"
            alt="Synthetic firm turnover distribution, weighted, £1,000 bins"
            width={2715}
            height={1603}
          />
        </div>
        <Caption className="mt-4 max-w-6xl">
          The near-threshold profile is {teal("calibrated to the OBR’s published £1k-band counts")}:
          density rises into the threshold and steps down across it — the administratively observed
          bunching shape, {teal("inherited from targets, not from firm behaviour")} (placebo later).
          The £150,000 step sits on an HMRC calibration-band edge; the seam at £65k is the edge of the published fine-band window.
        </Caption>
      </div>
    </Slide>
  );
}

/* ================================================================== */
/* 3 · Static costing                                                  */
/* ================================================================== */

export function SectionStatic() {
  return (
    <SectionSlide
      section="03 · Static costing"
      title="Static costing"
      subtitle="Mechanical reclassification on one common base — level, shape, and rate reforms."
    />
  );
}

export function StaticCostingSlide() {
  return (
    <Slide>
      <div className="grid h-full grid-cols-[1fr_1.1fr] items-center gap-12">
        <div className="p-2">
          <div className="text-2xl font-bold text-pe-dark">Static revenue</div>
          <div className="mt-8">
            <DisplayEq tex="R(T)=\sum_i w_i\,v_i\,\mathbf{1}\{y_i\ge T\}" className="text-[1.4rem]" />
          </div>
          <Caption className="mt-6">
            Recompute <Math tex="R" /> under each new schedule and difference against the base.
          </Caption>
        </div>
        <div>
          <SlideTitle kicker="Static costing">Mechanical reclassification</SlideTitle>
          <BList
            className="mt-6"
            size="text-lg"
            items={[
              <>
                Turnover held fixed <Math tex="\Rightarrow" /> revenue moves only through firms whose{" "}
                {teal("registration status flips")} (those between the old and new cutoff).
              </>,
              <>
                Three reform levers, four schedules, all on one base:
                <SubList
                  items={[
                    <><span className="font-semibold text-pe-dark">Level</span> — move the threshold up or down</>,
                    <><span className="font-semibold text-pe-dark">Shape</span> — graduated taper, 0% at £85k → 20% at £105k</>,
                    <><span className="font-semibold text-pe-dark">Rate</span> — reduced-rate band (10% or 15%) in £85k–£105k</>,
                  ]}
                />
              </>,
            ]}
          />
        </div>
      </div>
    </Slide>
  );
}

export function ValidationSlide() {
  return (
    <Slide>
      <div className="grid h-full grid-cols-[1.15fr_1fr] items-center gap-10">
        <Figure
          src="/figures/vat_threshold_revenue_impact.png"
          alt="April-2024 anchor reform: author estimate vs HMRC costing, 2024–25 to 2028–29"
          width={2639}
          height={1781}
        />
        <div>
          <SlideTitle kicker="Static costing">The April-2024 anchor reform</SlideTitle>
          <BList
            className="mt-6"
            size="text-lg"
            items={[
              <>
                Cost the <Math tex="\text{£}85\text{k}\to\text{£}90\text{k}" /> April-2024 rise, my
                estimate vs HMRC’s published costing — under two released-firm conventions.
              </>,
              <>
                Early years overshoot:{" "}
                {teal("2025–26: −£366m (all deregister) / −£209m (43% voluntary retention)")} vs
                HMRC’s −£185m — consistent with HMRC embedding a{" "}
                {teal("phased registration response")} (~28k fewer registrants in year 1).
              </>,
              <>
                Later years converge: {teal("2026–27: −£125.2m vs HMRC’s −£125m")} under retention;
                both conventions reproduce the 2028–29 sign flip via the{" "}
                {teal("uprated counterfactual threshold path")}.
              </>,
              <>
                Microdata are calibrated to HMRC aggregates <Math tex="\Rightarrow" /> a{" "}
                {teal("consistency check")}, not out-of-sample validation.
              </>,
            ]}
          />
        </div>
      </div>
    </Slide>
  );
}

export function StaticSweepSlide() {
  return (
    <Slide>
      <div className="flex h-full flex-col">
        <SlideTitle kicker="Static costing · 2025–26">
          Static sweep: cost of moving the threshold
        </SlideTitle>
        <Caption className="mt-3 max-w-6xl text-sm">
          Validated on the one move HMRC published, {teal("now sweep the whole schedule")} — the same
          mechanical reclassification at every threshold.
        </Caption>
        <div className="mt-3 grid min-h-0 flex-1 grid-cols-2 grid-rows-1 gap-6">
          <Figure
            src="/figures/revenue_impact_2025_26.png"
            alt="Revenue impact of moving the VAT threshold, 2025–26, vs £90k baseline"
            width={2674}
            height={1618}
          />
          <Figure
            src="/figures/firms_impact_2025_26.png"
            alt="Change in the number of VAT-paying firms across the threshold sweep"
            width={2639}
            height={1603}
          />
        </div>
        <Caption className="mt-3 max-w-6xl text-sm">
          From a £90k base (£204.6bn, 2024–25 vintage), roughly linear,{" "}
          {teal("≈£230–260m per £5k step")}: e.g. {teal("−£490m")} to raise to £100k,{" "}
          {teal("+£1,020m")} to lower to £70k. A £30k rise to £120k costs ~£1.4bn — 0.7% of the
          base.
        </Caption>
      </div>
    </Slide>
  );
}

export function ReformMenuSlide() {
  const rows: { reform: string; lever: string; stat: string; region: ReactNode; hi?: boolean }[] = [
    { reform: "Raise to £100k", lever: "Level", stat: "−£783m", region: "relocated" },
    {
      reform: "Taper £85–105k",
      lever: "Shape",
      stat: "−£550m",
      region: <span className="font-bold text-pe-teal">removed</span>,
      hi: true,
    },
    { reform: "Reduced 10%", lever: "Rate", stat: "−£497m", region: "split in two" },
    { reform: "Reduced 15%", lever: "Rate", stat: "−£248m", region: "split in two" },
  ];
  return (
    <Slide>
      <div className="flex h-full flex-col justify-center">
        <SlideTitle kicker="Static costing">The reform menu</SlideTitle>
        <div className="mt-8 overflow-hidden rounded-lg border border-slate-200 shadow-sm">
          <div className="grid grid-cols-[1.4fr_1fr_1fr_1.4fr] bg-pe-dark text-lg font-semibold text-white">
            {["Reform", "Lever", "Static", "Dominated region"].map((h, i) => (
              <div key={h} className={`px-6 py-4 ${i === 2 ? "text-right" : ""}`}>
                {h}
              </div>
            ))}
          </div>
          {rows.map((r) => (
            <div
              key={r.reform}
              className={`grid grid-cols-[1.4fr_1fr_1fr_1.4fr] border-t border-slate-200 text-xl ${
                r.hi ? "bg-pe-light" : "bg-white"
              }`}
            >
              <div className="px-6 py-4 font-semibold text-pe-dark">{r.reform}</div>
              <div className="px-6 py-4 text-slate-600">{r.lever}</div>
              <div className="px-6 py-4 text-right font-semibold tabular-nums text-slate-800">
                {r.stat}
              </div>
              <div className="px-6 py-4 text-slate-700">{r.region}</div>
            </div>
          ))}
        </div>
        <Caption className="mt-6 max-w-6xl">
          Common £85,000-notch / £184.8bn 2023–24 base (near-threshold profile calibrated to OBR
          £1k-band data).{" "}
          {teal("The level move carries the largest static cost and removes none of the dominated region")};
          the taper removes all of it (−£550m, against the 10% band’s −£497m). “Split in two” = a
          second notch appears where the reduced-rate band ends (£105k reverts to 20%), so the empty
          band is fragmented, not shrunk.
        </Caption>
        <SeeAppendix to={SLIDE.appBehav}>behavioural costs across e — appendix</SeeAppendix>
      </div>
    </Slide>
  );
}

/* ================================================================== */
/* 4 · The notch's distortion                                          */
/* ================================================================== */

export function SectionDistortion() {
  return (
    <SectionSlide
      section="04 · The distortion"
      title="The notch’s distortion"
      subtitle="The elasticity-free dominated region — and which reforms actually remove it."
    />
  );
}

export function BunchingMechanicalSlide() {
  return (
    <Slide>
      <div className="grid h-full grid-cols-[1fr_1fr] items-center gap-10">
        <div className="flex min-h-0 flex-col justify-center gap-2 self-stretch">
          <div className="min-h-0 flex-1">
            <Figure
              src="/figures/bunching_analysis_85k.png"
              alt="2023-24 population at £85k: no excess mass over the counterfactual"
              width={2611}
              height={1790}
            />
          </div>
          <Caption className="flex-none text-center">
            2023–24 at £85k: {teal("E = 14,663 — inherited from the OBR targets")}
          </Caption>
          <div className="min-h-0 flex-1">
            <Figure
              src="/figures/bunching_analysis_90k.png"
              alt="2024-25 population at £90k: a large spurious excess-mass signal at the band edge"
              width={2611}
              height={1790}
            />
          </div>
          <Caption className="flex-none text-center">
            2024–25 at £90k: {teal("E ≈ 196,000 — spurious")}, on an HMRC band edge
          </Caption>
        </div>
        <div>
          <SlideTitle kicker="The distortion">
            The near-threshold shape is target-inherited — provably
          </SlideTitle>
          <BList
            className="mt-6"
            size="text-lg"
            items={[
              <>
                The 2023–24 file reproduces the administrative bunching profile{" "}
                {teal("because it is calibrated to it")} (OBR £1k-band targets); the estimator
                reads back <Math tex="E=14{,}663" /> (b = 0.109).
              </>,
              <>
                {teal("Placebo")}: regenerate {teal("without")} the fine targets{" "}
                <Math tex="\Rightarrow E = 0" /> — the estimator reads targets, not firms (the
                generator has no location choice).
              </>,
              <>
                The uncalibrated 2024–25 vintage shows the reverse: {teal("a spurious E ≈ 196,000")}{" "}
                exactly on the £90k band edge, robust in all 16 degree×window cells.
              </>,
              <>
                {teal("Power test")}: injected true bunching {teal("is")} detected (55–82%
                recovered, attenuated, never inflated).
              </>,
              <>
                An earlier draft manufactured the step via a{" "}
                {teal("liability mis-scaling")} — corrected and documented openly. No behavioural
                claim is made from synthetic data; the administrative fact remains Liu et al.
                (2021).
              </>,
            ]}
          />
        </div>
      </div>
    </Slide>
  );
}

export function DominatedRegionSlide() {
  return (
    <Slide>
      <div className="grid h-full grid-cols-[1fr_1fr] items-center gap-10">
        <div className="flex h-full flex-col justify-center">
          <div className="min-h-0 flex-1">
            <Figure
              src="/figures/dynamic_notch_fit_e017.png"
              alt="The £85k notch and its dominated region (£85,000, £106,250)"
              width={2714}
              height={1846}
            />
          </div>
          <div className="mt-3">
            <Caption className="text-center">The notch in the firm’s profit:</Caption>
            <DisplayEq
              tex="\pi(y)=\begin{cases} y-c(y), & y\lt T^{*}\\ (1-\tau)\,y-c(y), & y\ge T^{*}\end{cases}"
              className="mt-2 text-[1rem]"
            />
          </div>
        </div>
        <div>
          <SlideTitle kicker="The distortion">
            The dominated region: an elasticity-free distortion
          </SlideTitle>
          <BList
            className="mt-6"
            size="text-lg"
            items={[
              <>
                At <Math tex="T^{*}" />, profit jumps <em>down</em> by{" "}
                <Math tex="\tau T^{*}=\text{£}17{,}000" /> — VAT on turnover already earned.
              </>,
              <>
                The {teal("dominated region")} is the band just above <Math tex="T^{*}" /> where no
                firm locates: <Math tex="a = T^{*}\dfrac{\tau}{1-\tau} =" /> {teal("£21,250")}.
              </>,
              <>Interval (£85,000, £106,250), ≈160,000 weighted firms.</>,
              <>
                Depends on <Math tex="\tau" /> and <Math tex="T^{*}" /> {teal("alone")} — no
                elasticity, and {teal("invariant to the firm’s input share")}: under the value-added
                formulation the deductible share cancels. An order of magnitude wider than a kink’s
                ~£1k.
              </>,
            ]}
          />
          <SeeAppendix to={SLIDE.appSecondary}>secondary-notch arithmetic — appendix</SeeAppendix>
        </div>
      </div>
    </Slide>
  );
}

export function WhichReformsSlide() {
  return (
    <Slide>
      <div className="flex h-full flex-col justify-center">
        <SlideTitle kicker="The distortion">Which reforms actually shrink the band?</SlideTitle>
        <BList
          className="mt-8"
          items={[
            <>
              <span className="font-semibold text-pe-dark">Raise the threshold</span> —{" "}
              {teal("relocates")} the £21,250 band unchanged to the new cutoff.
            </>,
            <>
              <span className="font-semibold text-pe-dark">Banded reduced rate</span> — does{" "}
              <em>not</em> shrink it. Lowers the single notch (£17,000 → £12,750 at 15%) but adds a{" "}
              {teal("secondary notch")} where the band reverts at £105k:
              <div className="my-3">
                <DisplayEq
                  tex="a' = T_1\dfrac{\tau-r}{1-\tau};\qquad a+a' = \text{£}21{,}562\ (15\%),\ \text{£}22{,}569\ (10\%)"
                  className="text-[1.05rem]"
                />
              </div>
              <span className="text-lg text-slate-500">
                It splits one wide empty band into two, not shrinks it.
              </span>
            </>,
            <>
              <span className="font-semibold text-pe-dark">Graduated taper</span> — continuous, no
              notch and no dominated region: {teal("removes the distortion entirely")}.
            </>,
          ]}
        />
        <p className="mt-8 text-center text-2xl font-semibold text-pe-dark">
          Only the taper removes the dominated region (−£550m); the 10% band costs −£497m and
          removes none of it.
        </p>
      </div>
    </Slide>
  );
}

/* ================================================================== */
/* 5 · Behavioural layer                                               */
/* ================================================================== */

export function SectionBehavioural() {
  return (
    <SectionSlide
      section="05 · Behavioural layer"
      title="Behavioural layer"
      subtitle="Re-optimising each firm under an assumed turnover elasticity — a sensitivity range, not an estimate."
    />
  );
}

export function FirmProblemSlide() {
  return (
    <Slide>
      <div className="grid h-full grid-cols-[1fr_1fr] items-center gap-10">
        <div>
          <SlideTitle kicker="Behavioural layer">
            The firm’s problem: VAT on <em className="italic">value added</em>
          </SlideTitle>
          <p className="mt-6 text-lg leading-snug text-slate-700">
            A firm of ability <Math tex="n" /> chooses turnover <Math tex="y" /> to maximise
          </p>
          <div className="my-4">
            <DisplayEq
              tex="\pi(y)=(1-\delta)\bigl(1-\tau f(y)\bigr)\,y-C(y;n,e)"
              className="text-[1.15rem]"
            />
          </div>
          <BList
            size="text-lg"
            items={[
              <>
                VAT falls on {teal("value added")} <Math tex="(1-\delta)y" />, not whole turnover;{" "}
                <Math tex="\delta" /> is the firm’s deductible-input share.
              </>,
              <>
                <Math tex="C" /> is the firm’s own-factor cost (labour, capital): inputs{" "}
                <Math tex="\delta y" /> are already netted by <Math tex="(1-\delta)" />.
              </>,
              <>
                Low <Math tex="\delta" /> — labour-intensive services (consultants, hairdressers,
                tutors): value added ≈ turnover. High <Math tex="\delta" /> — retail, wholesale,
                construction: large deductible input purchases.
              </>,
              <>
                Optimum <Math tex="y^{\star}=n\bigl[(1-\delta)(1-\tau)\bigr]^{e}" />; in the reform
                response ratio {teal("δ cancels")}, so the intensive response needs only{" "}
                <Math tex="e" /> and the schedule. Per-firm <Math tex="\delta_i" /> is drawn (mean VA
                share 40%); grounding it in ONS Supply–Use tables by sector is future work.
              </>,
            ]}
          />
        </div>
        <div className="flex h-full flex-col justify-center">
          <div className="min-h-0 flex-1">
            <Figure
              src="/figures/formulation_a_optima.png"
              alt="Optima of the value-added formulation-A profit under a hard notch vs a graduated taper"
              width={3863}
              height={1628}
            />
          </div>
          <Caption className="mt-3 text-center">
            Higher <Math tex="\delta" /> scales the curve down and pulls the registered optimum toward
            the threshold (eventually bunching).
          </Caption>
        </div>
      </div>
    </Slide>
  );
}

export function BehaviouralLayerSlide() {
  return (
    <Slide>
      <div className="grid h-full grid-cols-[1fr_1fr] items-center gap-10">
        <div className="flex h-full flex-col justify-center">
          <div className="min-h-0 flex-1">
            <Figure
              src="/figures/dynamic_cost_vs_elasticity.png"
              alt="Behavioural reform cost versus assumed elasticity for the three flat-rate reforms"
              width={2671}
              height={1927}
            />
          </div>
          <div className="mt-3">
            <Caption className="text-center">Each firm rescales turnover under the reform:</Caption>
            <DisplayEq
              tex="y_i^{\star}=y_i\left(\dfrac{1-\tau_{\text{reform}}}{1-\tau_{\text{base}}}\right)^{\!e}"
              className="mt-2 text-[1rem]"
            />
          </div>
        </div>
        <div>
          <SlideTitle kicker="Behavioural layer">
            The intensive margin barely moves the costings
          </SlideTitle>
          <BList
            className="mt-6"
            size="text-lg"
            items={[
              <>
                Region-confined iso-elastic simulator (value-added base, formulation A) re-optimises
                each firm within its schedule region, with <Math tex="e" /> {teal("assumed")}, not
                identified; sweep <Math tex="e\in\{0.05,0.17,0.32\}" />.
              </>,
              <>
                {teal("A level rise has exactly zero intensive-margin offset at every e")}: released
                firms leave the VAT base (their expansion is untaxed), crossers optimally bunch —
                so −£783m static {teal("is")} the behavioural cost.
              </>,
              <>
                Reduced-rate offsets are {teal("second order")} (&lt;4% even at{" "}
                <Math tex="e=0.32" />):
                <SubList
                  items={[
                    <>10% band: £497m → <span className="font-semibold text-pe-teal">£487m</span> at e = 0.17</>,
                    <>15% band: £248m → <span className="font-semibold text-pe-teal">£241m</span> at e = 0.17</>,
                  ]}
                />
              </>,
              <>
                Static cost is the exact <Math tex="e\to0" /> limit; <Math tex="e=0.05" /> is the
                external anchor (Kleven and Waseem, 2013), Liu et al.’s UK estimates (~0.09–0.14)
                fall in the swept range.
              </>,
              <>
                Taper excluded (continuously varying rate — outside the flat-rate machinery). The
                behavioural uncertainty that matters is {teal("extensive")}: registration and
                location choice, which these data cannot identify.
              </>,
            ]}
          />
        </div>
      </div>
    </Slide>
  );
}

/* ================================================================== */
/* 6 · Conclusion                                                      */
/* ================================================================== */

export function SectionConclusion() {
  return (
    <SectionSlide
      section="06 · Conclusion"
      title="Conclusion"
      subtitle="What is robust, what is conditional, and what comes next."
    />
  );
}

export function ConclusionSlide() {
  return (
    <Slide>
      <div className="flex h-full flex-col justify-center">
        <SlideTitle kicker="Conclusion">Conclusion</SlideTitle>
        <div className="mt-8 grid grid-cols-2 gap-12">
          <div>
            <div className="mb-4 text-sm font-bold uppercase tracking-[0.18em] text-pe-teal">
              What we find
            </div>
            <BList
              size="text-base"
              items={[
                <>
                  <span className="font-semibold text-pe-dark">What this is:</span> an open,
                  reproducible firm-level microsimulation of the UK VAT threshold as a turnover-tax
                  notch; prices level / shape / rate reforms {teal("three ways")}.
                </>,
                <>
                  <span className="font-semibold text-pe-dark">Static:</span> mechanical conventions
                  overshoot HMRC’s anchor early and {teal("match it by 2026–27 (−£125.2m vs −£125m)")},
                  reproducing the sign flip; raising to £100k costs £783m, the taper £550m.
                </>,
                <>
                  <span className="font-semibold text-pe-dark">Distortion:</span> the dominated region{" "}
                  <Math tex="a=\text{£}21{,}250" /> is elasticity-free and input-share invariant —{" "}
                  {teal("only the taper removes it")}; the level move relocates it, a reduced rate
                  splits it in two.
                </>,
                <>
                  <span className="font-semibold text-pe-dark">Behavioural:</span> the intensive
                  margin barely moves the costings — zero offset for a level rise, &lt;4% for the
                  bands; conditional on assumed <Math tex="e" />.
                </>,
                <>
                  <span className="font-semibold text-pe-dark">Transparency:</span> the
                  near-threshold shape is {teal("calibrated to published OBR £1k-band data")};
                  placebo and power tests show the estimator reads targets, not behaviour — in both
                  directions (spurious signal at the uncalibrated £90k band edge).
                </>,
              ]}
            />
          </div>
          <div>
            <div className="mb-4 text-sm font-bold uppercase tracking-[0.18em] text-pe-amber">
              Limitations & future work
            </div>
            <BList
              size="text-base"
              items={[
                <>
                  <span className="font-semibold text-pe-dark">Scope</span> — no firm location
                  choice; ~43% voluntary registration (Liu et al., 2021) enters only as a rate and
                  a retention sensitivity; net-repayment traders unmodelled;{" "}
                  {teal("sector-grounded δ from ONS Supply–Use to come")}.
                </>,
                <>
                  <span className="font-semibold text-pe-dark">Partial efficiency metric</span> —
                  dominated-region width indexes misallocation only; a full welfare / MVPF accounting
                  (Keen and Mintz, 2004) is needed.
                </>,
                <>
                  <span className="font-semibold text-pe-dark">Aggregate only</span> — no sectoral
                  incidence (VAT liability by sector excluded from the optimiser).
                </>,
                <>
                  <span className="font-semibold text-pe-dark">Link to household microsimulation</span>{" "}
                  — connect the firm-side VAT model to PolicyEngine’s household tax-benefit model to
                  trace incidence and distributional effects onto households.
                </>,
              ]}
            />
          </div>
        </div>
      </div>
    </Slide>
  );
}

export function ThankYouSlide() {
  return (
    <Slide isEnd>
      <div className="relative z-10 flex h-full flex-col items-center justify-center space-y-8 text-center">
        <Image
          alt="PolicyEngine"
          className="opacity-100"
          height={100}
          priority
          src="/logos/white.svg"
          style={{ height: "auto" }}
          width={340}
        />
        <div className="h-1 w-20 rounded-full bg-white/30" />
        <h1 className="font-display text-7xl font-bold leading-tight text-white">Thank you</h1>
        <div className="space-y-2 text-2xl text-white/85">
          <p>policyengine.org</p>
          <p>vahid@policyengine.org</p>
        </div>
        <div className="flex flex-col items-center gap-3">
          <div className="rounded-xl bg-white p-3 shadow-lg">
            <Image
              alt="QR code linking to the paper, data and code"
              src="/qr-paper.png"
              width={660}
              height={660}
              className="h-40 w-40"
            />
          </div>
          <p className="text-sm uppercase tracking-[0.14em] text-white/60">
            Scan for the paper, data &amp; code
          </p>
        </div>
      </div>
    </Slide>
  );
}

/* ================================================================== */
/* Appendix (linear, after the main deck)                              */
/* ================================================================== */

export function AppendixNotchSlide() {
  return (
    <Slide>
      <div className="flex h-full flex-col justify-center">
        <SlideTitle kicker="Appendix">Notch vs. kink</SlideTitle>
        <BList
          className="mt-8"
          items={[
            <>
              A <span className="font-semibold text-pe-dark">kink</span> taxes only the turnover{" "}
              <em>above</em> the threshold — the marginal rate jumps, but liability is continuous.
            </>,
            <>
              A <span className="font-semibold text-pe-dark">notch</span> taxes the firm’s{" "}
              <em>entire</em> turnover once it registers — liability jumps discretely by{" "}
              <Math tex="\tau T^{*}=\text{£}17{,}000" />.
            </>,
            <>
              The UK VAT threshold is a notch <Math tex="\Rightarrow" /> a strict{" "}
              {teal("dominated region")} exists, whereas a kink leaves only a ~£1k smoothed gap.
            </>,
            <>This is why the distortion can be measured exactly and without any elasticity.</>,
          ]}
        />
        <BackTo to={SLIDE.whyThreshold}>back to “Why the VAT threshold matters”</BackTo>
      </div>
    </Slide>
  );
}

export function AppendixGenSlide() {
  return (
    <Slide>
      <div className="flex h-full flex-col justify-center">
        <SlideTitle kicker="Appendix">How the synthetic firms are built</SlideTitle>
        <div className="mt-6 space-y-4">
          <p className="text-lg leading-snug text-slate-700">
            Draw firms by sector × ONS turnover band; turnover smoothed within band:
          </p>
          <DisplayEq
            tex="y_i=\underline{b}+(\overline{b}-\underline{b})\,u_i+\varepsilon_i,\quad u_i\sim\mathcal{U}(0,1),\ \varepsilon_i\sim\mathcal{N}(0,\sigma_b^2)"
            className="text-[1.05rem]"
          />
          <BList
            size="text-lg"
            items={[
              <>
                Input share <Math tex="x_i=\rho_i y_i" /> (<Math tex="\rho_i\in[0.1,0.95]" />,
                rescaled Beta, mean 0.6) <Math tex="\Rightarrow" /> net VAT liability{" "}
                <Math tex="v_i=0.20\,(y_i-x_i)" /> — the standard rate on value added.
              </>,
              <>
                Near-threshold £1k bins (£65k–£90k) target the OBR Chart C profile: direct counts
                above the threshold, shape (window-normalised) below it.
              </>,
              <>
                Each firm gets a positive weight <Math tex="w_i=e^{\theta_i}" />; weighted target{" "}
                <Math tex="\widehat{T}_k(\theta)=\sum_i a_{ki}w_i" />.
              </>,
            ]}
          />
          <p className="text-lg leading-snug text-slate-700">
            Fit <Math tex="\theta" /> by gradient descent on a symmetric relative-error loss:
          </p>
          <DisplayEq
            tex="L(\theta)=\tfrac{1}{K}\sum_k\lambda_k\min\{(\widehat{T}_k/T_k-1)^2,(T_k/\widehat{T}_k-1)^2\}+\tfrac{\alpha}{N}\sum_i|\theta_i|"
            className="text-[1.05rem]"
          />
          <p className="text-lg leading-snug text-slate-700">
            {teal("The threshold is a generator parameter")}: every counterfactual regenerates a
            fresh, internally consistent population.
          </p>
        </div>
        <BackTo to={SLIDE.data}>back to “Data”</BackTo>
      </div>
    </Slide>
  );
}

export function AppendixBehavSlide() {
  const headers = [
    <>Reform</>,
    <><Math tex="\text{Static }(e{=}0)" /></>,
    <><Math tex="e{=}0.05" /></>,
    <><Math tex="e{=}0.17" /></>,
    <><Math tex="e{=}0.32" /></>,
  ];
  const rows = [
    ["Raise to £100k", "−£783m", "−£783m", "−£783m", "−£783m"],
    ["Reduced 10% band", "−£497m", "−£494m", "−£487m", "−£479m"],
    ["Reduced 15% band", "−£248m", "−£246m", "−£241m", "−£234m"],
  ];
  return (
    <Slide>
      <div className="flex h-full flex-col justify-center">
        <SlideTitle kicker="Appendix">Behavioural costs across e</SlideTitle>
        <div className="mt-8 overflow-hidden rounded-lg border border-slate-200 shadow-sm">
          <div className="grid grid-cols-[1.5fr_1fr_1fr_1fr_1fr] bg-pe-dark text-lg font-semibold text-white">
            {headers.map((h, i) => (
              <div key={i} className={`px-5 py-4 ${i === 0 ? "" : "text-right"}`}>
                {h}
              </div>
            ))}
          </div>
          {rows.map((r) => (
            <div
              key={r[0]}
              className="grid grid-cols-[1.5fr_1fr_1fr_1fr_1fr] border-t border-slate-200 bg-white text-xl"
            >
              {r.map((c, i) => (
                <div
                  key={i}
                  className={`px-5 py-4 ${
                    i === 0
                      ? "font-semibold text-pe-dark"
                      : "text-right tabular-nums text-slate-800"
                  }`}
                >
                  {c}
                </div>
              ))}
            </div>
          ))}
        </div>
        <Caption className="mt-6 max-w-6xl">
          The raise is {teal("invariant in e")} — released firms leave the VAT base, so their
          expansion is untaxed and crossers optimally bunch below the new threshold. Band offsets
          stay under 4% (a larger <Math tex="e" /> makes them slightly cheaper). The taper is
          omitted — its rate varies continuously, outside the flat-rate machinery. The{" "}
          <Math tex="e\to0" /> limit reproduces the static costs {teal("exactly")} (asserted in
          code to £0.1m).
        </Caption>
        <BackTo to={SLIDE.reformMenu}>back to “The reform menu”</BackTo>
      </div>
    </Slide>
  );
}

export function AppendixSecondarySlide() {
  return (
    <Slide>
      <div className="flex h-full flex-col justify-center">
        <SlideTitle kicker="Appendix">Secondary-notch arithmetic (reduced rate)</SlideTitle>
        <p className="mt-6 text-xl text-slate-700">
          A reduced rate <Math tex="r" /> in <Math tex="[T^{*},T_1]" /> with{" "}
          <Math tex="T_1=\text{£}105{,}000" />:
        </p>
        <BList
          className="mt-6"
          items={[
            <>
              <span className="font-semibold text-pe-dark">Primary</span> notch shrinks: jump{" "}
              <Math tex="rT^{*}" />, width <Math tex="a=T^{*}\dfrac{r}{1-r}=" /> £15,000 (15%), £9,444
              (10%).
            </>,
            <>
              <span className="font-semibold text-pe-dark">Secondary</span> notch at the band top:
              jump <Math tex="(\tau-r)T_1=\text{£}5{,}250" /> (15%), width{" "}
              <Math tex="a'=T_1\dfrac{\tau-r}{1-\tau}=" /> £6,562 (15%), £13,125 (10%).
            </>,
            <>
              <span className="font-semibold text-pe-dark">Total</span> dominated width{" "}
              <Math tex="a+a'=\text{£}21{,}562" /> (15%), <Math tex="\text{£}22{,}569" /> (10%) — both
              marginally <em>above</em> the £21,250 baseline.
            </>,
          ]}
        />
        <p className="mt-8 text-center text-2xl font-semibold text-pe-dark">
          The reduced rate splits the empty band; it does not shrink it.
        </p>
        <BackTo to={SLIDE.dominated}>back to “The dominated region”</BackTo>
      </div>
    </Slide>
  );
}
