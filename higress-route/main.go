package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

func main() {
	// kubeconfig 默认路径
	var kubeconfig *string
	if home := homedir.HomeDir(); home != "" {
		kubeconfig = flag.String("kubeconfig", filepath.Join(home, ".kube", "config"), "(optional) absolute path to the kubeconfig file")
	} else {
		kubeconfig = flag.String("kubeconfig", "", "absolute path to the kubeconfig file")
	}

	// 命令行参数
	name := flag.String("name", "my-higress-route", "Ingress name")
	namespace := flag.String("namespace", "default", "Kubernetes namespace")
	host := flag.String("host", "api.example.com", "Host header")
	path := flag.String("path", "/v1/llm", "URL path")
	serviceName := flag.String("service", "vllm-service", "Backend service name")
	servicePort := flag.Int("port", 8000, "Backend service port")

	flag.Parse()

	// 创建 Kubernetes 客户端
	config, err := clientcmd.BuildConfigFromFlags("", *kubeconfig)
	if err != nil {
		panic(err.Error())
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err.Error())
	}

	// 创建 Ingress 路由
	opts := IngressOptions{
		Name:        *name,
		Namespace:   *namespace,
		Host:        *host,
		Path:        *path,
		PathType:    networkingv1.PathTypePrefix,
		ServiceName: *serviceName,
		ServicePort: int32(*servicePort),
	}

	err = CreateIngressRoute(clientset, opts)
	if err != nil {
		fmt.Printf("创建路由失败: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Higress 路由创建成功!")
	fmt.Printf("  名称: %s\n", *name)
	fmt.Printf("  命名空间: %s\n", *namespace)
	fmt.Printf("  主机: %s\n", *host)
	fmt.Printf("  路径: %s\n", *path)
	fmt.Printf("  后端服务: %s:%d\n", *serviceName, *servicePort)
}
