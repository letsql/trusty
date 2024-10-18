{
  description = "A devShell with Crane for Cargo builds for trusty";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix.url = "github:nix-community/poetry2nix";
  };

  outputs = { nixpkgs, rust-overlay, crane, flake-utils, poetry2nix, ... }:
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
        
        debug = msg: value:
          builtins.trace "${msg}: ${builtins.toJSON value}" value;
      
        allowedExtensions = [
          "csv"
          "json"
        ];
      
        hasAllowedExtension = path:
          let
            extension = pkgs.lib.lists.last (pkgs.lib.strings.splitString "." (baseNameOf (toString path)));
          in
          debug "hasAllowedExtension" (pkgs.lib.lists.any (ext: ext == extension) allowedExtensions);
      
        customFilter = path: type:
          let
            isCargoSource = craneLib.filterCargoSources path type;
            isAllowed = type == "regular" && hasAllowedExtension path;
            result = isCargoSource || isAllowed;
          in
          debug "customFilter ${toString path}" result;

        commonArgs = {
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

        trustyClippy = craneLib.cargoClippy (commonArgs // {
          inherit cargoArtifacts;
          cargoClippyExtraArgs = "--all-targets -- --deny warnings";
        });

        trustyTests = craneLib.cargoTest (commonArgs // {
          inherit cargoArtifacts;
        });
        
        poetryApplication = pkgs.poetry2nix.mkPoetryApplication {
          projectDir = ./.;
          preferWheels = true;
          overrides = pkgs.poetry2nix.overrides.withDefaults
            (self: super: {
              atpublic = super.atpublic.overridePythonAttrs
              (
                old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [super.hatchling];
                }
              );
              xgboost = super.xgboost.overridePythonAttrs (old: {
              } // pkgs.lib.attrsets.optionalAttrs pkgs.stdenv.isDarwin {
                nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ super.cmake ];
                cmakeDir = "../cpp_src";
                preBuild = ''
                  cd ..
                '';
              });
            });
        };

        pythonEnv = poetryApplication.dependencyEnv;

      in
      {
        packages = {
          default = trusty;
          pyApp = poetryApplication;
        };

        checks = {
          inherit
            trusty
            trustyClippy
            trustyTests;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ trusty ];
          buildInputs = [
            rustToolchain
            pythonEnv
            pkgs.poetry
          ];
        };
      });
}
