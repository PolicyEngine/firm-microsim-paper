import Slide from "@/components/core/Slide";
import Image from "@/components/core/BasePathImage";
import { formatDate, SpeakerInfo } from "@/lib/types";

interface CoverSlideProps {
  title: string;
  subtitle?: string;
  event: string;
  date: string;
  speakers: SpeakerInfo[];
}

export default function CoverSlide({
  title,
  subtitle,
  event,
  date,
  speakers,
}: CoverSlideProps) {
  return (
    <Slide isCover>
      <div className="relative z-10 flex h-full flex-col items-center justify-center space-y-9 text-center">
        <Image
          alt="PolicyEngine"
          className="opacity-100"
          height={100}
          priority
          src="/logos/white.svg"
          style={{ height: "auto" }}
          width={350}
        />

        <div className="h-1 w-20 rounded-full bg-white/30" />

        <h1 className="max-w-6xl text-center font-display text-6xl font-bold leading-tight text-white">
          {title}
        </h1>

        {subtitle && (
          <p className="max-w-4xl text-center text-2xl leading-relaxed text-white/80">
            {subtitle}
          </p>
        )}

        <div className="mt-2 flex flex-wrap items-center justify-center gap-8">
          {speakers.map((speaker) => (
            <div key={speaker.name} className="flex items-center justify-center gap-6">
              {speaker.headshot ? (
                <Image
                  alt={speaker.name}
                  className="h-24 w-24 rounded-full border-2 border-white/40 object-cover"
                  height={96}
                  src={speaker.headshot}
                  width={96}
                />
              ) : (
                <div className="flex h-24 w-24 items-center justify-center rounded-full border-2 border-white/40 bg-white/10 text-3xl font-bold">
                  {speaker.name
                    .split(" ")
                    .map((part) => part[0])
                    .join("")}
                </div>
              )}
              <div className="text-left">
                <p className="text-2xl font-semibold text-white">{speaker.name}</p>
                <p className="text-lg font-light text-white/70">{speaker.title}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="pt-2 text-center text-white opacity-60">
          <p>{event}</p>
          <p>{formatDate(date)}</p>
        </div>
      </div>
    </Slide>
  );
}
