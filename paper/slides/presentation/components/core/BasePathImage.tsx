import Image, { ImageProps } from "next/image";
import { resolveImageSrc } from "@/lib/base-path-image";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";

export default function BasePathImage({ alt, src, ...props }: ImageProps) {
  const resolvedSrc = typeof src === "string" ? resolveImageSrc(src, BASE_PATH) : src;
  return <Image {...props} alt={alt} src={resolvedSrc} />;
}
