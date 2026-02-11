/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'ai-blue': '#2563eb',
        'ai-purple': '#7c3aed',
        'ai-green': '#059669',
        'academy-primary': '#1e40af',
        'academy-secondary': '#f59e0b',
      },
      fontFamily: {
        'academy': ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}