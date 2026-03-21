package main

import (
	"context"
	"fmt"

	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// IngressOptions 定义创建 Ingress 路由的选项
type IngressOptions struct {
	Name        string
	Namespace   string
	Host        string
	Path        string
	PathType    networkingv1.PathType
	ServiceName string
	ServicePort int32
	Annotations map[string]string
}

// CreateIngressRoute 使用 Ingress API 创建 Higress 路由
func CreateIngressRoute(client *kubernetes.Clientset, opts IngressOptions) error {
	// 默认注解
	annotations := map[string]string{
		"kubernetes.io/ingress.class": "higress",
	}

	// 合并用户自定义注解
	if opts.Annotations != nil {
		for k, v := range opts.Annotations {
			annotations[k] = v
		}
	}

	ingress := &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:        opts.Name,
			Namespace:   opts.Namespace,
			Annotations: annotations,
		},
		Spec: networkingv1.IngressSpec{
			Rules: []networkingv1.IngressRule{
				{
					Host: opts.Host,
					IngressRuleValue: networkingv1.IngressRuleValue{
						HTTP: &networkingv1.HTTPIngressRuleValue{
							Paths: []networkingv1.HTTPIngressPath{
								{
									Path:     opts.Path,
									PathType: &opts.PathType,
									Backend: networkingv1.IngressBackend{
										Service: &networkingv1.IngressServiceBackend{
											Name: opts.ServiceName,
											Port: networkingv1.ServiceBackendPort{
												Number: opts.ServicePort,
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	created, err := client.NetworkingV1().Ingresses(opts.Namespace).Create(context.TODO(), ingress, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	fmt.Printf("创建的 Ingress: %s/%s\n", created.Namespace, created.Name)
	return nil
}

// DeleteIngressRoute 删除 Ingress 路由
func DeleteIngressRoute(client *kubernetes.Clientset, namespace, name string) error {
	return client.NetworkingV1().Ingresses(namespace).Delete(context.TODO(), name, metav1.DeleteOptions{})
}

// ListIngressRoutes 列出指定命名空间的所有 Ingress 路由
func ListIngressRoutes(client *kubernetes.Clientset, namespace string) error {
	list, err := client.NetworkingV1().Ingresses(namespace).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return err
	}

	fmt.Printf("\n=== %s 命名空间中的 Ingress 列表 ===\n", namespace)
	for _, ingress := range list.Items {
		fmt.Printf("- %s (Host: %s)\n", ingress.Name, ingress.Spec.Rules[0].Host)
	}
	fmt.Println()

	return nil
}

// GetIngressRoute 获取指定的 Ingress 路由详情
func GetIngressRoute(client *kubernetes.Clientset, namespace, name string) error {
	ingress, err := client.NetworkingV1().Ingresses(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return err
	}

	fmt.Printf("\n=== Ingress 详情: %s/%s ===\n", namespace, name)
	fmt.Printf("Host: %s\n", ingress.Spec.Rules[0].Host)
	fmt.Printf("Annotations:\n")
	for k, v := range ingress.Annotations {
		fmt.Printf("  %s: %s\n", k, v)
	}
	fmt.Println()

	return nil
}

// CreateIngressRouteWithAdvancedConfig 使用高级配置创建 Higress 路由
func CreateIngressRouteWithAdvancedConfig(client *kubernetes.Clientset, opts IngressOptions) error {
	// Higress 高级注解
	annotations := map[string]string{
		"kubernetes.io/ingress.class": "higress",
		// 超时配置
		"nginx.ingress.kubernetes.io/proxy-connect-timeout": "600",
		"nginx.ingress.kubernetes.io/proxy-send-timeout":    "600",
		"nginx.ingress.kubernetes.io/proxy-read-timeout":   "600",
		// 重试配置
		"nginx.ingress.kubernetes.io/proxy-next-upstream":      "error,timeout,http_502,http_503,http_504",
		"nginx.ingress.kubernetes.io/proxy-next-upstream-tries": "3",
		// CORS
		"nginx.ingress.kubernetes.io/enable-cors":      "true",
		"nginx.ingress.kubernetes.io/cors-allow-origin": "*",
		// 限流
		"nginx.ingress.kubernetes.io/limit-rps": "100",
	}

	// 合并用户自定义注解
	if opts.Annotations != nil {
		for k, v := range opts.Annotations {
			annotations[k] = v
		}
	}

	opts.Annotations = annotations
	return CreateIngressRoute(client, opts)
}

// CreateIngressRouteWithCanary 创建支持金丝雀发布的 Ingress 路由
func CreateIngressRouteWithCanary(client *kubernetes.Clientset, opts IngressOptions, canaryConfig CanaryConfig) error {
	annotations := map[string]string{
		"kubernetes.io/ingress.class": "higress",
		// 金丝雀配置
		"nginx.ingress.kubernetes.io/canary":                   "true",
		"nginx.ingress.kubernetes.io/canary-by-header":          canaryConfig.ByHeader,
		"nginx.ingress.kubernetes.io/canary-by-header-value":    canaryConfig.ByHeaderValue,
		"nginx.ingress.kubernetes.io/canary-weight":             fmt.Sprintf("%d", canaryConfig.Weight),
	}

	if opts.Annotations != nil {
		for k, v := range opts.Annotations {
			annotations[k] = v
		}
	}

	opts.Annotations = annotations
	return CreateIngressRoute(client, opts)
}

// CanaryConfig 金丝雀发布配置
type CanaryConfig struct {
	ByHeader      string
	ByHeaderValue string
	Weight        int
}
