/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: {
                    DEFAULT: '#6366f1',
                    dark: '#4f46e5',
                    light: '#818cf8',
                },
                secondary: '#ec4899',
                accent: '#06b6d4',
                blood: {
                    'a-plus': '#ef4444',
                    'a-minus': '#f97316',
                    'b-plus': '#3b82f6',
                    'b-minus': '#6366f1',
                    'ab-plus': '#8b5cf6',
                    'ab-minus': '#a855f7',
                    'o-plus': '#22c55e',
                    'o-minus': '#14b8a6',
                },
            },
            fontFamily: {
                sans: ['Inter', 'SF Pro Display', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
            },
            animation: {
                'float': 'float 6s ease-in-out infinite',
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            },
            backdropBlur: {
                xs: '2px',
            },
        },
    },
    plugins: [],
}
