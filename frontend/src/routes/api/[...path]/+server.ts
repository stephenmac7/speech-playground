import { json, type RequestHandler } from '@sveltejs/kit';
import { PYTHON_API_URL } from '$env/static/private';

const handler: RequestHandler = async ({ request, params, url }) => {
	const targetUrl = `${PYTHON_API_URL}/${params.path}${url.search}`;

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

export const GET = handler;
export const POST = handler;