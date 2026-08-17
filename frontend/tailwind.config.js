/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        legal: {
          bg: '#060A0E',
          header: '#0B1017',
          surface: '#0F161E',
          secondary: '#16202B',
          elevated: '#16202B',
          border: '#253444',
          text: '#F5F6F8',
          textSec: '#AAB4BE',
          textMuted: '#74808D',
          accent: '#5E84AC',
          accentLight: '#8EABC8',
          danger: '#D16B73',
          warning: '#D49A4A',
          success: '#55B18A',
          info: '#5E84AC',
          action: '#8EABC8',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
