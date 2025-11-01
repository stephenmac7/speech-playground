import { json } from '@sveltejs/kit';
import { ALIGNER_API_URL } from '$env/static/private';

// export async function GET() {
//     const pythonResponse = await fetch(PYTHON_API_URL, {
//         method: 'GET',
//         headers: {
//             'Content-Type': 'application/json',
//         }
//     }).catch((error) => {
//         console.error('Error connecting to Python service:', error);
//         return json({ detail: 'Error connecting to the processing service' }, { status: 502 }); // Bad Gateway
//     });
//     return json(await pythonResponse.json(), { status: pythonResponse.status });
// }

export async function POST({ request }) {
	const targetUrl = `${ALIGNER_API_URL}/align`;

	const headers = new Headers(request.headers);
	headers.delete('host'); // Let fetch set the correct host
	headers.delete('cookie'); // Don't forward the user's SvelteKit session cookie

	try {
		return await fetch(targetUrl, {
			method: request.method,
			headers: headers,
			body: request.body,
			duplex: 'half' // Required for streaming request bodies
		} as RequestInit);
	} catch (error) {
		console.error('Error connecting to Python service:', error);
		return json({ detail: 'Error connecting to the processing service' }, { status: 502 }); // Bad Gateway
	}
}
