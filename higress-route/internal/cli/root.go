package cli

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	"path/filepath"
)

var (
	kubeconfig string
)

// NewRootCmd creates the root command for the CLI
func NewRootCmd() *cobra.Command {
	rootCmd := &cobra.Command{
		Use:   "hr",
		Short: "Higress Gateway API CLI tool",
		Long: `A command-line tool for managing Higress routes using Kubernetes Gateway API.

This tool creates, lists, and deletes HTTPRoute resources that are
associated with Higress Gateway.`,
	}

	// Global flag for kubeconfig
	if home := homedir.HomeDir(); home != "" {
		rootCmd.PersistentFlags().StringVar(
			&kubeconfig,
			"kubeconfig",
			filepath.Join(home, ".kube", "config"),
			"(optional) absolute path to the kubeconfig file",
		)
	} else {
		rootCmd.PersistentFlags().StringVar(
			&kubeconfig,
			"kubeconfig",
			"",
			"absolute path to the kubeconfig file",
		)
	}

	// Add subcommands
	rootCmd.AddCommand(NewCreateCmd())
	rootCmd.AddCommand(NewDeleteCmd())
	rootCmd.AddCommand(NewListCmd())
	rootCmd.AddCommand(NewGetCmd())

	return rootCmd
}

// Execute runs the root command
func Execute() {
	if err := NewRootCmd().Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// getKubeconfigPath returns the kubeconfig path to use
func getKubeconfigPath() string {
	if kubeconfig != "" {
		return kubeconfig
	}
	if home := homedir.HomeDir(); home != "" {
		return filepath.Join(home, ".kube", "config")
	}
	// Try loading rules default
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	return loadingRules.GetDefaultFilename()
}
