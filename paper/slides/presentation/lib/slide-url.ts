export function buildSlideUrl(currentUrl: string, slide: number): string {
  const url = new URL(currentUrl);
  url.searchParams.set("slide", String(slide));
  return `${url.pathname}${url.search}${url.hash}`;
}
