import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatProbability(prob: string | number) {
  if (typeof prob === "string") return prob;
  return `${(prob * 100).toFixed(1)}%`;
}
