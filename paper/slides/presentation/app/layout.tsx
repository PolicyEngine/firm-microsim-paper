import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "A Firm-Level Microsimulation for VAT Policy Analysis",
  description:
    "PolicyEngine conference deck for the IMA World Congress 2026 — costing UK VAT registration-threshold reforms on synthetic firm data.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
