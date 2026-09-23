// Centralized API helpers for client-side requests
// Exposes typed helpers with consistent error handling.
import { base } from '$app/paths';

// Callers pass root-absolute paths like `/api/compare`. Prefixing `base` keeps
// those working when the app is mounted under a subdirectory rather than at the
// root of a host; `base` is '' when it isn't, so this is a no-op then.
const withBase = (url: string) => (url.startsWith('/') ? `${base}${url}` : url);

export async function postJson<T>(
	url: string,
	formData: FormData,
	signal?: AbortSignal
): Promise<T> {
	const res = await fetch(withBase(url), { method: 'POST', body: formData, signal });
	if (!res.ok) {
		let message = res.statusText;
		try {
			const data = await res.json();
			// SvelteKit error payload is often { message } or { detail }
			message = (data?.detail ?? data?.message ?? message) as string;
		} catch {
			// ignore JSON parse errors
		}
		throw new Error(message);
	}
	return (await res.json()) as T;
}

export async function postBlob(
	url: string,
	formData: FormData,
	signal?: AbortSignal
): Promise<Blob> {
	const res = await fetch(withBase(url), { method: 'POST', body: formData, signal });
	if (!res.ok) {
		let message = res.statusText;
		try {
			const data = await res.json();
			message = (data?.detail ?? data?.message ?? message) as string;
		} catch {
			// ignore
		}
		throw new Error(message);
	}
	return await res.blob();
}

export async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
	const res = await fetch(withBase(url), { method: 'GET', signal });
	if (!res.ok) {
		let message = res.statusText;
		try {
			const data = await res.json();
			message = (data?.detail ?? data?.message ?? message) as string;
		} catch {
			// ignore
		}
		throw new Error(message);
	}
	return (await res.json()) as T;
}

export async function getBlob(url: string, signal?: AbortSignal): Promise<Blob> {
	const res = await fetch(withBase(url), { method: 'GET', signal });
	if (!res.ok) {
		let message = res.statusText;
		try {
			const data = await res.json();
			message = (data?.detail ?? data?.message ?? message) as string;
		} catch {
			// ignore
		}
		throw new Error(message);
	}
	return await res.blob();
}
