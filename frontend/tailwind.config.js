/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        industrial: {
          bg: '#0B0F17',
          card: '#151C28',
          border: '#2A3447',
          accent: '#00E5FF',
        }
      }
    },
  },
  plugins: [],
}
