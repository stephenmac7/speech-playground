/**
 * Reports an error to the user and console.
 * @param description A short, user-friendly description of what went wrong.
 * @param error The original error object or message.
 */
export function reportError(description: string, error?: any) {
	alert(description);
	if (error) {
		console.error('An error occurred:', error);
	}
}
