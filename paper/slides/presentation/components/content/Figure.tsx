import Image from "@/components/core/BasePathImage";

interface FigureProps {
  src: string;
  alt: string;
  width: number;
  height: number;
}

export default function Figure({ src, alt, width, height }: FigureProps) {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <Image
        alt={alt}
        src={src}
        width={width}
        height={height}
        className="h-auto max-h-full w-full object-contain"
        style={{ height: "auto" }}
      />
    </div>
  );
}
