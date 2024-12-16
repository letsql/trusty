{
  description = "A devShell for poetry and cargo for trusty";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix.url = "github:nix-community/poetry2nix";
    pre-commit-hooks = {
      url = "github:cachix/pre-commit-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = { nixpkgs, rust-overlay, crane, flake-utils, poetry2nix, pre-commit-hooks, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [
          (import rust-overlay)
          poetry2nix.overlays.default
        ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default;
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        allowedExtensions = [
          "csv"
          "json"
        ];

        hasAllowedExtension = path:
          let
            extension = pkgs.lib.lists.last (pkgs.lib.strings.splitString "." (baseNameOf (toString path)));
          in
          pkgs.lib.lists.any (ext: ext == extension) allowedExtensions;

        customFilter = path: type:
          let
            isCargoSource = craneLib.filterCargoSources path type;
            isAllowed = type == "regular" && hasAllowedExtension path;
          in
          isCargoSource || isAllowed;
        cargoDeps = pkgs.rustPlatform.importCargoLock {
          lockFile = ./Cargo.lock;
          outputHashes = {
            "gbdt-0.1.3" = "sha256-f2uqulFSNGwrDM7RPdGIW11VpJRYexektXjHxTJHHmA=";
          };
        };
        commonArgs = {
          inherit cargoDeps;
          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter = customFilter;
          };
          strictDeps = true;
          buildInputs = with pkgs; [
            openssl
          ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
            pkgs.darwin.apple_sdk.frameworks.Security
          ];

          nativeBuildInputs = with pkgs; [
            pkg-config
          ];
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        trusty = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });

        datasets = {
          diamonds = pkgs.fetchurl {
            url = "https://raw.githubusercontent.com/tidyverse/ggplot2/master/data-raw/diamonds.csv";
            sha256 = "sha256-lXRzCwOrokHYmcSpdRHFBhsZNY+riVEHdPtsJBaDRcQ=";
          };

          airline = pkgs.fetchurl {
            url = "https://raw.githubusercontent.com/varundixit4/Airline-Passenger-Satisfaction-Report/refs/heads/main/airline_satisfaction.csv";
            sha256 = "sha256-oV+rbTamEj3tsDXhvBGzHye1R2cc6NJ3YudNllJ8Nk8=";
          };
        };

        datafusion-udf = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
          cargoExtraArgs = "--package examples --bin datafusion_udf";
          doCheck = false;
        });

        datafusion-udf-wrapper = pkgs.writeScriptBin "datafusion-udf" ''
          #!${pkgs.stdenv.shell}
          exec ${datafusion-udf}/bin/datafusion_udf "$@"
        '';

        dataFiles = pkgs.runCommand "trusty-data-files" { } ''
          mkdir -p $out/data
          ln -s ${datasets.diamonds} $out/data/diamonds.csv
          ln -s ${datasets.airline} $out/data/airline_satisfaction.csv
        '';

        poetryApplication = pkgs.poetry2nix.mkPoetryApplication {
          projectDir = ./.;
          preferWheels = true;
          python = pkgs.python312;
          src = pkgs.lib.cleanSourceWith {
            src = ./.;
            filter = customFilter;
          };
          cargoDeps = pkgs.rustPlatform.importCargoLock {
            lockFile = ./Cargo.lock;
            outputHashes = {
              "gbdt-0.1.3" = "sha256-f2uqulFSNGwrDM7RPdGIW11VpJRYexektXjHxTJHHmA=";
            };
          };
          nativeBuildInputs = [
            pkgs.rustPlatform.cargoSetupHook
            pkgs.rustPlatform.maturinBuildHook
            rustToolchain
            pkgs.pkg-config
          ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
          ];
          buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
          ];
          overrides = pkgs.poetry2nix.overrides.withDefaults
            (self: super: {
              xgboost = super.xgboost.overridePythonAttrs (old: { } // pkgs.lib.attrsets.optionalAttrs pkgs.stdenv.isDarwin {
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ super.cmake ];
                cmakeDir = "../cpp_src";
                preBuild = ''
                  cd ..
                '';
              });
            });
        };

        buildMaturinScript = pkgs.writeScriptBin "build-maturin" ''
          #!${pkgs.stdenv.shell}
          echo "Building maturin wheel..."
          maturin build
            
          WHEEL_PATH="target/wheels"
          WHEEL_FILE=$(ls ''${WHEEL_PATH}/*.whl | head -n 1)
          PYTHON_DIR="python/trusty"
          TMP_DIR="tmp_wheel"
            
          # Create temporary directory
          mkdir -p ''${TMP_DIR}
            
          # Unzip the wheel to temporary directory
          ${pkgs.unzip}/bin/unzip -q "''${WHEEL_FILE}" -d ''${TMP_DIR}
            
          # Find the .so file (works for both .so and .dylib)
          SO_FILE=$(find ''${TMP_DIR} -name "*.so" -o -name "*.dylib")
            
          if [ -z "''${SO_FILE}" ]; then
              echo "No .so or .dylib file found in wheel"
              exit 1
          fi
            
          # Create the destination directory if it doesn't exist
          mkdir -p "''${PYTHON_DIR}"
            
          # Copy the .so file to the Python package directory
          cp "''${SO_FILE}" "''${PYTHON_DIR}/"
            
          echo "Copied $(basename "''${SO_FILE}") to ''${PYTHON_DIR}/"
            
          # Clean up
          rm -rf ''${TMP_DIR}
        '';

        processScript = pkgs.writeScriptBin "prepare-benchmarks" ''
          #!${pkgs.stdenv.shell}
          
          mkdir -p data
          echo "Copying data files from Nix store..."
          cp -f ${dataFiles}/data/* data/
          ${poetryApplication}/bin/generate-examples --data_dir data --base_dir data --generation_type benchmark
        '';
        pythonEnv = poetryApplication.dependencyEnv;
        clippy-hook = pkgs.writeScript "clippy-hook" ''
          #!${pkgs.stdenv.shell}
          export CARGO_HOME="$PWD/.cargo"
          export RUSTUP_HOME="$PWD/.rustup"
          export PATH="${rustToolchain}/bin:$PATH"
          mkdir -p target
          exec ${rustToolchain}/bin/cargo clippy --all-targets --all-features -- -D warnings
        '';

        pre-commit-check = pre-commit-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            nixpkgs-fmt.enable = true;
            rustfmt.enable = true;
            ruff = {
              enable = true;
              files = "\\.py$";
              excludes = [ ];
            };
            clippy = {
              enable = true;
              entry = toString clippy-hook;
            };
          };
          tools = {
            ruff = pkgs.ruff;
            rustfmt = rustToolchain;
            clippy = rustToolchain;
          };
        };
      in
      {
        apps = rec {
          poetryApp = {
            type = "app";
            program = "${poetryApplication.dependencyEnv}/bin/ipython";
          };
          default = poetryApp;
        };
        packages = {
          trusty = trusty;
          pyApp = poetryApplication;
          data = dataFiles;
          datafusion-udf-example = datafusion-udf-wrapper;
          inherit poetryApplication;
          default = datafusion-udf-wrapper;
        };

        checks = {
          # primary issue was that `nix flake check` runs in a pure environment,
          # preventing Clippy and Cargo from accessing the internet or untracked files.
          # this caused failures when trying to fetch Git dependencies like `gbdt`.
          # more info: https://github.com/cachix/git-hooks.nix/issues/452
          # we replace direct pre-commit-hook with a custom mkDerivation

          pre-commit-check = pkgs.stdenv.mkDerivation {
            name = "pre-commit-check";
            src = ./.;

            nativeBuildInputs = [
              pkgs.rustPlatform.cargoSetupHook
              rustToolchain
            ];

            buildInputs = with pkgs; [
              git
              openssl
              pkg-config
              rustToolchain
            ];

            cargoDeps = pkgs.rustPlatform.importCargoLock {
              lockFile = ./Cargo.lock;
              outputHashes = {
                "gbdt-0.1.3" = "sha256-f2uqulFSNGwrDM7RPdGIW11VpJRYexektXjHxTJHHmA=";
              };
            };

            buildPhase = ''
              export CARGO_HOME="$PWD/.cargo"
              export RUSTUP_HOME="$PWD/.rustup"
              export PATH="${rustToolchain}/bin:$PATH"
              ${pre-commit-check.buildCommand}
            '';

            installPhase = ''
              touch $out
            '';

            RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          };
        };
        devShells.default = pkgs.mkShell {
          inputsFrom = [ trusty ];
          buildInputs = [
            rustToolchain
            pythonEnv
            pkgs.poetry
            pkgs.maturin
            # add pre-commit dependencies
            pkgs.ruff
            pkgs.rustfmt
            pkgs.nixpkgs-fmt
            processScript
            buildMaturinScript
          ];
          shellHook = ''
            ${pre-commit-check.shellHook}
            echo "Run 'build-maturin' to rebuild and install the package"
          '';
        };
        devShells.danShell = pkgs.mkShell {
          buildInputs = [
            poetryApplication
            rustToolchain
          ];
        };
      });
}
