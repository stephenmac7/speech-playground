// Shared frontend types mirroring backend /models response and local encoder config.

export type EncoderOption = {
	value: string;
	label: string;
	discretizers: Array<string>;
	default_dist_method: string;
	has_fixed_frame_rate: boolean;
	supports_dpdp: boolean;
};

export type VoiceModelOption = { value: string; label: string };

export type ModelsResponse = {
	encoders: Array<EncoderOption>;
	vc_models: Array<VoiceModelOption>;
};

export type EncoderConfig = {
	encoder: string;
	discretize: boolean;
	discretizer: string;
	dpdp: boolean;
	gamma: string;
};
