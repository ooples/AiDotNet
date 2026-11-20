export function serverRequest(value: string, options?: {method?:string;headers?:Record<string,string>;data?:unknown;timeout?:number}, fetcher?:typeof fetch): Promise<{
  status():number;ok():boolean;url():string;headers():Record<string,string>;text():Promise<string>;json():Promise<any>;dispose():Promise<void>;
}>;
