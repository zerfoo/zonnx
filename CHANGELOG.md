# Changelog

## [1.0.0](https://github.com/zerfoo/zonnx/compare/v0.9.0...v1.0.0) (2026-03-30)


### Bug Fixes

* **test:** correct GGUF magic constant in test assertions ([1c4b3a6](https://github.com/zerfoo/zonnx/commit/1c4b3a60809a8cee136b9f303b27f8729ee1efd9))


### Miscellaneous Chores

* release 1.0.0 ([5f32544](https://github.com/zerfoo/zonnx/commit/5f325440693b2f7261e18b3404630f3fd4085174))

## [0.9.0](https://github.com/zerfoo/zonnx/compare/v0.8.0...v0.9.0) (2026-03-26)


### Features

* **granite2gguf:** add Granite Time Series to GGUF converter ([3f7b701](https://github.com/zerfoo/zonnx/commit/3f7b7014bff23c48a819f3aabf72e7395ee7d621))

## [0.8.0](https://github.com/zerfoo/zonnx/compare/v0.7.0...v0.8.0) (2026-03-26)


### Features

* **safetensors:** add SafeTensors file reader ([d646bd8](https://github.com/zerfoo/zonnx/commit/d646bd8fcbc088a2b533394a6d10e0b84d4e6180))

## [0.7.0](https://github.com/zerfoo/zonnx/compare/v0.6.0...v0.7.0) (2026-03-21)


### Features

* **converter:** add safetensors-to-GGUF conversion for BERT models ([86f8918](https://github.com/zerfoo/zonnx/commit/86f89181a06ff82571f988518e46167640c27feb))
* **gguf:** add BERT/RoBERTa tensor name and metadata mapping ([349920f](https://github.com/zerfoo/zonnx/commit/349920fdae4b36ba8139f3abf0466b06197744e8))

## [0.6.0](https://github.com/zerfoo/zonnx/compare/v0.5.0...v0.6.0) (2026-03-16)


### Features

* **cli:** output GGUF instead of ZMF (ADR-037) ([4ddf906](https://github.com/zerfoo/zonnx/commit/4ddf90621bfe7ebf8aa50faf73ed26ddc34afb2d))
* **gguf:** add ONNX to GGUF metadata mapping ([ab6fa8e](https://github.com/zerfoo/zonnx/commit/ab6fa8ed002ed3636fe8cd96d6302f0498883db4))
* **gguf:** add ONNX to GGUF tensor name mapping ([eaa5aa1](https://github.com/zerfoo/zonnx/commit/eaa5aa1f5a32c6266937602118a53e1341342cfa))
* **gguf:** implement GGUF v3 binary writer ([8e7f233](https://github.com/zerfoo/zonnx/commit/8e7f233174dd00c1a380c0bb0010393cb67fa6f3))


### Bug Fixes

* update goreleaser config for v2.6+ and fix hardcoded test path ([0166e1c](https://github.com/zerfoo/zonnx/commit/0166e1c933754a1d3c1b259ec03bd29b5a29ceb3))

## [0.5.0](https://github.com/zerfoo/zonnx/compare/v0.4.0...v0.5.0) (2026-03-13)


### Features

* Add initial importer package from zerfoo ([1e89e21](https://github.com/zerfoo/zonnx/commit/1e89e219fe617584d2ce26cb98a110674a4f5024))
* add initial zonnx structure and native ONNX parser ([08b46c1](https://github.com/zerfoo/zonnx/commit/08b46c1498171e7d4d0e016dfe91909d5a0b2764))
* add zonnx CLI binary to Makefile build target ([721e2d6](https://github.com/zerfoo/zonnx/commit/721e2d64380776c5f8d9841f900562f136552373))
* **cli:** Add API key support to download command and update tests ([ff57f8a](https://github.com/zerfoo/zonnx/commit/ff57f8a02eae18e921c143a3113a0c267a77b0c6))
* **cli:** Add download subcommand to zonnx CLI and integration test ([2dabfb1](https://github.com/zerfoo/zonnx/commit/2dabfb1e9c89f44470128898a98def870a2e40f5))
* **cli:** Implement CLI UX for unified inspect command ([01b24ef](https://github.com/zerfoo/zonnx/commit/01b24eff71b50f0bbb059345d9cebaf738eb34fb))
* **downloader:** Add API key support and update tests ([dd4403d](https://github.com/zerfoo/zonnx/commit/dd4403d45c39823d9622cb99464b2e3a4f938e27))
* **downloader:** Define ModelSource interface and basic downloader structure ([d4bc150](https://github.com/zerfoo/zonnx/commit/d4bc1506c2dc2551ad1ac754dcf3d8279ee58967))
* **downloader:** Implement core download logic and add tests ([46f1dfd](https://github.com/zerfoo/zonnx/commit/46f1dfd2a6bf45bc006f58b64108c6780f7d8a54))
* **downloader:** Implement HuggingFaceSource and integrate tests into downloader_test.go ([448c050](https://github.com/zerfoo/zonnx/commit/448c050d889ac2291c841cbb13d6f9f5dbf298de))
* **importer:** Add converters for Reshape and Transpose ([dd75b01](https://github.com/zerfoo/zonnx/commit/dd75b01a3f97b72e78b7f7686096717621ee5e9f))
* **importer:** Implement basic ONNX to ZMF graph conversion ([65b6446](https://github.com/zerfoo/zonnx/commit/65b644685a7cbbc243b782134bfd7ca168f7d21e))
* **importer:** Implement Reshape converter ([3745138](https://github.com/zerfoo/zonnx/commit/374513873abd49ffb3ac50901026e58095d0625a))
* **importer:** Implement Transpose converter ([2334982](https://github.com/zerfoo/zonnx/commit/2334982917ef5fd69cb22fac735c6f546b979d7a))
* **inspector:** Implement basic ONNX and ZMF inspection functions and add tests ([98af07c](https://github.com/zerfoo/zonnx/commit/98af07cdad4af858e07ceb06d2dd199189895a77))


### Bug Fixes

* Apply formatting fixes to test_convert.go ([c98cdca](https://github.com/zerfoo/zonnx/commit/c98cdca5ce454436b9478b7a063326b6eb3e1e27))
* bump zmf dependency to v0.4.0 for quantization type support ([dff555b](https://github.com/zerfoo/zonnx/commit/dff555b2ade27909bf64abd71c405d6e7c418f15))
* cleanup ([f0338a7](https://github.com/zerfoo/zonnx/commit/f0338a7ea570a245d88be76a5c6d8d167d781e66))
* **cmd/zonnx:** code formatting ([f1fb12e](https://github.com/zerfoo/zonnx/commit/f1fb12e27d6dfaa2e967bcb98b7404a685214585))
* convert flags ([141908f](https://github.com/zerfoo/zonnx/commit/141908fcaa833cd00676a71b555c456cf82cd848))
* **converter:** Apply linting and formatting fixes ([69e3b30](https://github.com/zerfoo/zonnx/commit/69e3b30466a7cf81e25a06459aa97558b27d451b))
* **importer/layers:** Apply linting and formatting fixes to layer files ([091cca1](https://github.com/zerfoo/zonnx/commit/091cca17667351e19bc08848f9872edaf1cfd3cd))
* **importer:** Apply linting and formatting fixes to importer.go ([eb51263](https://github.com/zerfoo/zonnx/commit/eb51263e7becf1c16855e98aef6499aab56a4462))
* **importer:** Comment out failing test and remove unused imports ([7178011](https://github.com/zerfoo/zonnx/commit/7178011d6aa96e262624062cc416a441744f58c8))
* **onnx:** Apply linting and formatting fixes to generated protobuf file ([8a149bf](https://github.com/zerfoo/zonnx/commit/8a149bf0f446203a0105871a860d233e7fa66478))
* **pkg/downloader:** code formatting ([9f03502](https://github.com/zerfoo/zonnx/commit/9f035029f3a72ccd50dc932584824ef3038ded01))
* **pkg/importer:** code formatting ([efd1d8f](https://github.com/zerfoo/zonnx/commit/efd1d8f605225bd6d5b88133bed115f04ba32220))
* **pkg/inspector:** code formatting ([084cd5c](https://github.com/zerfoo/zonnx/commit/084cd5c08901b9aea4662f3e212a0b472600a2c0))
* **pkg/registry:** code formatting ([dadc3fe](https://github.com/zerfoo/zonnx/commit/dadc3fea001cf54c5bb97777a6c15bdd7b4d6a59))
* skip goreleaser validation for dirty state from replace directive removal ([9691775](https://github.com/zerfoo/zonnx/commit/9691775c53e8825cdb8798e576809bbe2fdeb583))
* **zmf_inspector:** Replace ioutil with os package ([7dde31e](https://github.com/zerfoo/zonnx/commit/7dde31e1f1c5ecf79ebeabadbf7980d8dc26d4c6))

## [0.4.0](https://github.com/zerfoo/zonnx/compare/v0.3.0...v0.4.0) (2026-03-13)


### Features

* Add initial importer package from zerfoo ([1e89e21](https://github.com/zerfoo/zonnx/commit/1e89e219fe617584d2ce26cb98a110674a4f5024))
* add initial zonnx structure and native ONNX parser ([08b46c1](https://github.com/zerfoo/zonnx/commit/08b46c1498171e7d4d0e016dfe91909d5a0b2764))
* add zonnx CLI binary to Makefile build target ([721e2d6](https://github.com/zerfoo/zonnx/commit/721e2d64380776c5f8d9841f900562f136552373))
* **cli:** Add API key support to download command and update tests ([ff57f8a](https://github.com/zerfoo/zonnx/commit/ff57f8a02eae18e921c143a3113a0c267a77b0c6))
* **cli:** Add download subcommand to zonnx CLI and integration test ([2dabfb1](https://github.com/zerfoo/zonnx/commit/2dabfb1e9c89f44470128898a98def870a2e40f5))
* **cli:** Implement CLI UX for unified inspect command ([01b24ef](https://github.com/zerfoo/zonnx/commit/01b24eff71b50f0bbb059345d9cebaf738eb34fb))
* **downloader:** Add API key support and update tests ([dd4403d](https://github.com/zerfoo/zonnx/commit/dd4403d45c39823d9622cb99464b2e3a4f938e27))
* **downloader:** Define ModelSource interface and basic downloader structure ([d4bc150](https://github.com/zerfoo/zonnx/commit/d4bc1506c2dc2551ad1ac754dcf3d8279ee58967))
* **downloader:** Implement core download logic and add tests ([46f1dfd](https://github.com/zerfoo/zonnx/commit/46f1dfd2a6bf45bc006f58b64108c6780f7d8a54))
* **downloader:** Implement HuggingFaceSource and integrate tests into downloader_test.go ([448c050](https://github.com/zerfoo/zonnx/commit/448c050d889ac2291c841cbb13d6f9f5dbf298de))
* **importer:** Add converters for Reshape and Transpose ([dd75b01](https://github.com/zerfoo/zonnx/commit/dd75b01a3f97b72e78b7f7686096717621ee5e9f))
* **importer:** Implement basic ONNX to ZMF graph conversion ([65b6446](https://github.com/zerfoo/zonnx/commit/65b644685a7cbbc243b782134bfd7ca168f7d21e))
* **importer:** Implement Reshape converter ([3745138](https://github.com/zerfoo/zonnx/commit/374513873abd49ffb3ac50901026e58095d0625a))
* **importer:** Implement Transpose converter ([2334982](https://github.com/zerfoo/zonnx/commit/2334982917ef5fd69cb22fac735c6f546b979d7a))
* **inspector:** Implement basic ONNX and ZMF inspection functions and add tests ([98af07c](https://github.com/zerfoo/zonnx/commit/98af07cdad4af858e07ceb06d2dd199189895a77))


### Bug Fixes

* Apply formatting fixes to test_convert.go ([c98cdca](https://github.com/zerfoo/zonnx/commit/c98cdca5ce454436b9478b7a063326b6eb3e1e27))
* cleanup ([f0338a7](https://github.com/zerfoo/zonnx/commit/f0338a7ea570a245d88be76a5c6d8d167d781e66))
* **cmd/zonnx:** code formatting ([f1fb12e](https://github.com/zerfoo/zonnx/commit/f1fb12e27d6dfaa2e967bcb98b7404a685214585))
* convert flags ([141908f](https://github.com/zerfoo/zonnx/commit/141908fcaa833cd00676a71b555c456cf82cd848))
* **converter:** Apply linting and formatting fixes ([69e3b30](https://github.com/zerfoo/zonnx/commit/69e3b30466a7cf81e25a06459aa97558b27d451b))
* **importer/layers:** Apply linting and formatting fixes to layer files ([091cca1](https://github.com/zerfoo/zonnx/commit/091cca17667351e19bc08848f9872edaf1cfd3cd))
* **importer:** Apply linting and formatting fixes to importer.go ([eb51263](https://github.com/zerfoo/zonnx/commit/eb51263e7becf1c16855e98aef6499aab56a4462))
* **importer:** Comment out failing test and remove unused imports ([7178011](https://github.com/zerfoo/zonnx/commit/7178011d6aa96e262624062cc416a441744f58c8))
* **onnx:** Apply linting and formatting fixes to generated protobuf file ([8a149bf](https://github.com/zerfoo/zonnx/commit/8a149bf0f446203a0105871a860d233e7fa66478))
* **pkg/downloader:** code formatting ([9f03502](https://github.com/zerfoo/zonnx/commit/9f035029f3a72ccd50dc932584824ef3038ded01))
* **pkg/importer:** code formatting ([efd1d8f](https://github.com/zerfoo/zonnx/commit/efd1d8f605225bd6d5b88133bed115f04ba32220))
* **pkg/inspector:** code formatting ([084cd5c](https://github.com/zerfoo/zonnx/commit/084cd5c08901b9aea4662f3e212a0b472600a2c0))
* **pkg/registry:** code formatting ([dadc3fe](https://github.com/zerfoo/zonnx/commit/dadc3fea001cf54c5bb97777a6c15bdd7b4d6a59))
* skip goreleaser validation for dirty state from replace directive removal ([9691775](https://github.com/zerfoo/zonnx/commit/9691775c53e8825cdb8798e576809bbe2fdeb583))
* **zmf_inspector:** Replace ioutil with os package ([7dde31e](https://github.com/zerfoo/zonnx/commit/7dde31e1f1c5ecf79ebeabadbf7980d8dc26d4c6))

## [0.3.0](https://github.com/zerfoo/zonnx/compare/v0.2.0...v0.3.0) (2026-03-13)


### Features

* Add initial importer package from zerfoo ([1e89e21](https://github.com/zerfoo/zonnx/commit/1e89e219fe617584d2ce26cb98a110674a4f5024))
* add initial zonnx structure and native ONNX parser ([08b46c1](https://github.com/zerfoo/zonnx/commit/08b46c1498171e7d4d0e016dfe91909d5a0b2764))
* add zonnx CLI binary to Makefile build target ([721e2d6](https://github.com/zerfoo/zonnx/commit/721e2d64380776c5f8d9841f900562f136552373))
* **cli:** Add API key support to download command and update tests ([ff57f8a](https://github.com/zerfoo/zonnx/commit/ff57f8a02eae18e921c143a3113a0c267a77b0c6))
* **cli:** Add download subcommand to zonnx CLI and integration test ([2dabfb1](https://github.com/zerfoo/zonnx/commit/2dabfb1e9c89f44470128898a98def870a2e40f5))
* **cli:** Implement CLI UX for unified inspect command ([01b24ef](https://github.com/zerfoo/zonnx/commit/01b24eff71b50f0bbb059345d9cebaf738eb34fb))
* **downloader:** Add API key support and update tests ([dd4403d](https://github.com/zerfoo/zonnx/commit/dd4403d45c39823d9622cb99464b2e3a4f938e27))
* **downloader:** Define ModelSource interface and basic downloader structure ([d4bc150](https://github.com/zerfoo/zonnx/commit/d4bc1506c2dc2551ad1ac754dcf3d8279ee58967))
* **downloader:** Implement core download logic and add tests ([46f1dfd](https://github.com/zerfoo/zonnx/commit/46f1dfd2a6bf45bc006f58b64108c6780f7d8a54))
* **downloader:** Implement HuggingFaceSource and integrate tests into downloader_test.go ([448c050](https://github.com/zerfoo/zonnx/commit/448c050d889ac2291c841cbb13d6f9f5dbf298de))
* **importer:** Add converters for Reshape and Transpose ([dd75b01](https://github.com/zerfoo/zonnx/commit/dd75b01a3f97b72e78b7f7686096717621ee5e9f))
* **importer:** Implement basic ONNX to ZMF graph conversion ([65b6446](https://github.com/zerfoo/zonnx/commit/65b644685a7cbbc243b782134bfd7ca168f7d21e))
* **importer:** Implement Reshape converter ([3745138](https://github.com/zerfoo/zonnx/commit/374513873abd49ffb3ac50901026e58095d0625a))
* **importer:** Implement Transpose converter ([2334982](https://github.com/zerfoo/zonnx/commit/2334982917ef5fd69cb22fac735c6f546b979d7a))
* **inspector:** Implement basic ONNX and ZMF inspection functions and add tests ([98af07c](https://github.com/zerfoo/zonnx/commit/98af07cdad4af858e07ceb06d2dd199189895a77))


### Bug Fixes

* Apply formatting fixes to test_convert.go ([c98cdca](https://github.com/zerfoo/zonnx/commit/c98cdca5ce454436b9478b7a063326b6eb3e1e27))
* cleanup ([f0338a7](https://github.com/zerfoo/zonnx/commit/f0338a7ea570a245d88be76a5c6d8d167d781e66))
* **cmd/zonnx:** code formatting ([f1fb12e](https://github.com/zerfoo/zonnx/commit/f1fb12e27d6dfaa2e967bcb98b7404a685214585))
* convert flags ([141908f](https://github.com/zerfoo/zonnx/commit/141908fcaa833cd00676a71b555c456cf82cd848))
* **converter:** Apply linting and formatting fixes ([69e3b30](https://github.com/zerfoo/zonnx/commit/69e3b30466a7cf81e25a06459aa97558b27d451b))
* **importer/layers:** Apply linting and formatting fixes to layer files ([091cca1](https://github.com/zerfoo/zonnx/commit/091cca17667351e19bc08848f9872edaf1cfd3cd))
* **importer:** Apply linting and formatting fixes to importer.go ([eb51263](https://github.com/zerfoo/zonnx/commit/eb51263e7becf1c16855e98aef6499aab56a4462))
* **importer:** Comment out failing test and remove unused imports ([7178011](https://github.com/zerfoo/zonnx/commit/7178011d6aa96e262624062cc416a441744f58c8))
* **onnx:** Apply linting and formatting fixes to generated protobuf file ([8a149bf](https://github.com/zerfoo/zonnx/commit/8a149bf0f446203a0105871a860d233e7fa66478))
* **pkg/downloader:** code formatting ([9f03502](https://github.com/zerfoo/zonnx/commit/9f035029f3a72ccd50dc932584824ef3038ded01))
* **pkg/importer:** code formatting ([efd1d8f](https://github.com/zerfoo/zonnx/commit/efd1d8f605225bd6d5b88133bed115f04ba32220))
* **pkg/inspector:** code formatting ([084cd5c](https://github.com/zerfoo/zonnx/commit/084cd5c08901b9aea4662f3e212a0b472600a2c0))
* **pkg/registry:** code formatting ([dadc3fe](https://github.com/zerfoo/zonnx/commit/dadc3fea001cf54c5bb97777a6c15bdd7b4d6a59))
* skip goreleaser validation for dirty state from replace directive removal ([9691775](https://github.com/zerfoo/zonnx/commit/9691775c53e8825cdb8798e576809bbe2fdeb583))
* **zmf_inspector:** Replace ioutil with os package ([7dde31e](https://github.com/zerfoo/zonnx/commit/7dde31e1f1c5ecf79ebeabadbf7980d8dc26d4c6))

## [0.2.0](https://github.com/zerfoo/zonnx/compare/v0.1.0...v0.2.0) (2026-03-13)


### Features

* add zonnx CLI binary to Makefile build target ([721e2d6](https://github.com/zerfoo/zonnx/commit/721e2d64380776c5f8d9841f900562f136552373))
