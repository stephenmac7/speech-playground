// src/lib/db.ts
import { PUBLIC_EXAMPLE_PATH } from '$env/static/public';
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

db.on('populate', async () => {
	const count = await db.audio_tracks.count();
	if (count === 0) {
		console.log('No audio tracks found, pre-initializing database');
		let blob: Blob | undefined;
		try {
			blob = await getBlob(`/api/data/${PUBLIC_EXAMPLE_PATH}`);
			console.log('Fetched example audio blob');
		} catch (e) {
			console.error('Error fetching example audio:', e);
		}
		await db.audio_tracks.add({
			keys: ['Audio'],
			data: blob
		});
	}
});

export type { AudioTrack };
export { db };
