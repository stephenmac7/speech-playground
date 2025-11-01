// Centralized API helpers for client-side requests
// Exposes typed helpers with consistent error handling.

export async function postJson<T>(
	url: string,
	formData: FormData,
	signal?: AbortSignal
): Promise<T> {
	const res = await fetch(url, { method: 'POST', body: formData, signal });
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
	const res = await fetch(url, { method: 'POST', body: formData, signal });
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
	const res = await fetch(url, { method: 'GET', signal });
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
	const res = await fetch(url, { method: 'GET', signal });
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
