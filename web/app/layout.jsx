import "./globals.css";

export const metadata = {
  title: "Taş Kağıt Makas AI",
  description: "Elini göster, ne yaptığını tahmin edeyim. Model tamamen tarayıcında çalışır.",
};
export const viewport = { width: "device-width", initialScale: 1, themeColor: "#eef1f5" };

export default function RootLayout({ children }) {
  return (
    <html lang="tr">
      <body>{children}</body>
    </html>
  );
}
