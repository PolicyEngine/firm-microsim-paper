export function resolveImageSrc(src: string, basePath: string): string {
  if (!basePath) return src;
  if (src.startsWith("/") && !src.startsWith(`${basePath}/`)) {
    return `${basePath}${src}`;
  }
  return src;
}
