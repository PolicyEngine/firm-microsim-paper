import Slide from "@/components/core/Slide";

interface SectionSlideProps {
  section: string;
  title: string;
  subtitle: string;
}

export default function SectionSlide({ section, title, subtitle }: SectionSlideProps) {
  return (
    <Slide className="bg-pe-light">
      <div className="flex h-full items-center">
        <div>
          <div className="mb-5 text-sm font-bold uppercase tracking-[0.24em] text-pe-teal">
            {section}
          </div>
          <h1 className="max-w-5xl text-6xl font-extrabold leading-tight tracking-tight text-pe-dark">
            {title}
          </h1>
          <p className="mt-8 max-w-4xl text-2xl leading-snug text-slate-600">
            {subtitle}
          </p>
        </div>
      </div>
    </Slide>
  );
}
