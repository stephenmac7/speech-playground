// src/lib/db.ts
import { PUBLIC_EXAMPLE_PATH, PUBLIC_EXAMPLE_MODEL_PATH } from '$env/static/public';
import { getBlob } from '$lib/api';
import Dexie, { type EntityTable } from 'dexie';

interface AudioTrack {
	id: number;
	keys: string[];
	data?: Blob;
}

const db = new Dexie('AudioDatabase') as Dexie & {
	audio_tracks: EntityTable<AudioTrack, 'id'>;
};

db.version(1).stores({
	audio_tracks: '++id, *keys'
});

db.on('ready', async (db) => {
	return db.table('audio_tracks').count(async (count) => {
		if (count === 0) {
			console.log('No audio tracks found, pre-initializing database');
			try {
				const [audio, model] = await Promise.all([
					getBlob(`/api/data/${PUBLIC_EXAMPLE_PATH}`),
					getBlob(`/api/data/${PUBLIC_EXAMPLE_MODEL_PATH}`)
				]);
				console.log('Fetched example audio and model blobs');
				await Promise.all([
					db.table('audio_tracks').add({ keys: ['Audio'], data: audio }),
					db.table('audio_tracks').add({ keys: ['Model'], data: model })
				]);
				console.log('Done populating.');
			} catch (e) {
				console.error('Error fetching or populating examples:', e);
			}
		}
	});
});

export type { AudioTrack };
export { db };
